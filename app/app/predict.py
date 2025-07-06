import sys
import os
import chess.pgn
import pandas as pd
import joblib
import re
import ast
from tqdm import tqdm

def process_pgn(pgn_file):
    pgn = open(pgn_file)
    data = []
    while True:
        try:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            headers = game.headers
            data.append([
                headers.get("Event", "?"), headers.get("White", "?"), headers.get("Black", "?"),
                headers.get("Result", "?"), headers.get("UTCDate", "?"), headers.get("UTCTime", "?"),
                headers.get("WhiteElo", "?"), headers.get("BlackElo", "?"),
                headers.get("WhiteRatingDiff", "?"), headers.get("BlackRatingDiff", "?"),
                headers.get("ECO", "?"), headers.get("Opening", "?"), headers.get("TimeControl", "?"),
                headers.get("Termination", "?")
            ])
        except Exception as e:
            print(f"Error al leer una partida: {e}")
            break
    return pd.DataFrame(data, columns=[
        "Event", "White", "Black", "Result", "UTCDate", "UTCTime", "WhiteElo", "BlackElo",
        "WhiteRatingDiff", "BlackRatingDiff", "ECO", "Opening", "TimeControl", "Termination"
    ])

def extraer_movimientos_desde_pgn(pgn_path):
    with open(pgn_path, 'r') as file:
        lines = file.readlines()
        movimientos = "".join(lines[16:]).strip()
    return movimientos

def extraer_jugadas_san(anotacion):
    limpio = re.sub(r'\{[^}]*\}', '', anotacion)
    limpio = re.sub(r'\d+\.', '', limpio)
    limpio = re.sub(r'\s+', ' ', limpio).strip()
    limpio = re.sub(r'\s*(1-0|0-1|1/2-1/2)\s*$', '', limpio)
    jugadas = limpio.split(' ')
    return jugadas

def load_evaluaciones(log_file_path):
    with open(log_file_path, 'r') as file:
        log_content = file.read()
    eval_values = re.findall(r'[\+\-]?\d+[,\.]\d{2}', log_content)
    return [float(value.replace(',', '.')) for value in eval_values]

def vectorizar_fen(fen, piezas):
    board = chess.Board(fen)
    vector = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            val = piezas[piece.piece_type]
            val = val if piece.color == chess.WHITE else -val
        else:
            val = 0
        vector.append(val)
    return vector

def extraer_features_fen(fen):
    board = chess.Board(fen)
    valor_piezas = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    material_blancas = sum(valor_piezas[p.piece_type] for p in board.piece_map().values() if p.color)
    material_negras = sum(valor_piezas[p.piece_type] for p in board.piece_map().values() if not p.color)
    return {
        'material_blancas': material_blancas,
        'material_negras': material_negras,
        'diferencia_material': material_blancas - material_negras,
        'turno': int(board.turn),
        'enroque_blancas': int(board.has_kingside_castling_rights(chess.WHITE) or board.has_queenside_castling_rights(chess.WHITE)),
        'enroque_negras': int(board.has_kingside_castling_rights(chess.BLACK) or board.has_queenside_castling_rights(chess.BLACK)),
        'jaque': int(board.is_check()),
        'mate': int(board.is_checkmate())
    }

def clasificar_jugada(eval_actual, eval_anterior):
    if pd.isna(eval_actual) or pd.isna(eval_anterior):
        return 'desconocida'
    delta = eval_actual - eval_anterior
    if abs(delta) < 0.15:
        return 'buena'
    elif delta < -0.15:
        return 'error'
    else:
        return 'dudosa'

def main(pgn_path, log_path, model_path):
    df = process_pgn(pgn_path)
    movimientos = extraer_movimientos_desde_pgn(pgn_path)
    df['AN'] = [movimientos for _ in range(len(df))]
    df['SAN'] = df['AN'].apply(extraer_jugadas_san)
    df['ECO_Family'] = df['ECO'].str[0]
    df['Result'] = df['Result'].map({'1-0': 1, '1/2-1/2': 0, '0-1': -1})
    
    # 2. Evaluaciones
    evals = load_evaluaciones(log_path)
    df['evals'] = [evals for _ in range(len(df))]
    
    # 3.Preprocesamiento para calculo posiciones FEN para computer vision
    datos_fen = []
    piezas = {None: 0, chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3, chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6}
    for idx, row in df.iterrows():
        board = chess.Board()
        jugadas = row['SAN']
        evaluaciones = row['evals']
        for i, jugada in enumerate(jugadas):
            try:
                board.push_san(jugada)
                eval_actual = evaluaciones[i] if i < len(evaluaciones) else None
                f = extraer_features_fen(board.fen())
                datos_fen.append({
                    'ID_partida': idx,
                    'jugada_num': i + 1,
                    'SAN': jugada,
                    'eval': eval_actual,
                    'FEN': board.fen(),
                    'ECO_Family': row['ECO_Family'],
                    'Result': row['Result'],
                    'White': row['White'],
                    'Black': row['Black'],
                    'WhiteElo': row['WhiteElo'],
                    'BlackElo': row['BlackElo'],
                    **f
                })
            except Exception:
                break
    df_fens = pd.DataFrame(datos_fen)
    
    # Preprocesamiento de posiciones vectorizadas (sin uso porque era para computer vision)
    vector_fens = df_fens['FEN'].apply(lambda fen: vectorizar_fen(fen, piezas))
    fen_vector_df = pd.DataFrame(vector_fens.tolist(), columns=[f'FEN_{i}' for i in range(64)])
    df_fens = pd.concat([df_fens.reset_index(drop=True), fen_vector_df], axis=1)
    
    # 5. Preprocesamiento clasificación jugada
    df_fens['eval_anterior'] = df_fens.groupby('ID_partida')['eval'].shift(1)
    df_fens['calidad_jugada'] = df_fens.apply(lambda row: clasificar_jugada(row['eval'], row['eval_anterior']), axis=1)
    
    # Preprocesamiento de portcetaje de jugadas
    resumen = (
        df_fens.groupby('ID_partida')['calidad_jugada']
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .rename(columns={'buena': 'pct_buenas', 'dudosa': 'pct_dudosas', 'error': 'pct_errores'})
    )
    for col in ['pct_buenas', 'pct_dudosas', 'pct_errores']:
        if col not in resumen.columns:
            resumen[col] = 0.0
    df_ml = df_fens.merge(resumen, on='ID_partida', how='left')
    
    # ^reprocesamiento de datos
    df_ml['eval_anterior'] = df_ml['eval_anterior'].fillna(0)
    df_ml['eval'] = df_ml['eval'].fillna(0)
    exclude_cols = [col for col in df_ml.columns if col.startswith('FEN_')]
    exclude_cols += ['BlackElo', 'ID_partida', 'jugada_num', 'SAN', 'White', 'Black', 'FEN', 'WhiteElo', 'Result']
    exclude_cols += ['material_blancas', 'material_negras', 'diferencia_material', 'turno', 'enroque_blancas', 'enroque_negras', 'jaque', 'mate']
    df_ml = df_ml.drop(exclude_cols, axis=1, errors='ignore')
    categorical_cols = ['ECO_Family', 'calidad_jugada']
    df_ml = pd.get_dummies(df_ml, columns=[col for col in categorical_cols if col in df_ml.columns], drop_first=True)
    
    # Carga swl modelo 
    modelo_rf = joblib.load(model_path)
    columnas_esperadas = modelo_rf.feature_names_in_
    for col in columnas_esperadas:
        if col not in df_ml.columns:
            df_ml[col] = 0
    df_ml = df_ml[columnas_esperadas]
    predicciones_elo = modelo_rf.predict(df_ml)
    df_ml['prediccion_elo_negro'] = predicciones_elo
    media_elo_negro = round(df_ml['prediccion_elo_negro'].mean())
    print(f"El ELO del jugador negro es: {media_elo_negro}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Uso: python predict.py archivo.pgn archivo.log modelo.joblib")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
