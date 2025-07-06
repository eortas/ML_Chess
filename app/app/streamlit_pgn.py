import streamlit as st
import chess.pgn
import pandas as pd
import joblib
import re
import tempfile
import os
from tqdm import tqdm
import chess

# Configuración de la página
st.set_page_config(
    page_title="Chess ELO Predictor",
    page_icon="♟️",
    layout="wide"
)

def process_pgn(pgn_content):
    """Procesa el contenido PGN desde texto"""
    data = []
    games = pgn_content.split('\n\n')
    
    for game_text in games:
        if not game_text.strip():
            continue
            
        try:
            # Crear un objeto StringIO para simular un archivo
            from io import StringIO
            pgn_io = StringIO(game_text)
            game = chess.pgn.read_game(pgn_io)
            
            if game is None:
                continue
                
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
            st.warning(f"Error al leer una partida: {e}")
            continue
            
    return pd.DataFrame(data, columns=[
        "Event", "White", "Black", "Result", "UTCDate", "UTCTime", "WhiteElo", "BlackElo",
        "WhiteRatingDiff", "BlackRatingDiff", "ECO", "Opening", "TimeControl", "Termination"
    ])

def extraer_movimientos_desde_pgn(pgn_content):
    """Extrae movimientos del contenido PGN"""
    lines = pgn_content.split('\n')
    # Buscar donde terminan los headers y empiezan los movimientos
    start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('[') and ']' in line:
            start_idx = i + 1
        elif line.strip() and not line.startswith('['):
            start_idx = i
            break
    
    movimientos = " ".join(lines[start_idx:]).strip()
    return movimientos

def extraer_jugadas_san(anotacion):
    """Extrae jugadas en notación SAN"""
    limpio = re.sub(r'\{[^}]*\}', '', anotacion)
    limpio = re.sub(r'\d+\.', '', limpio)
    limpio = re.sub(r'\s+', ' ', limpio).strip()
    limpio = re.sub(r'\s*(1-0|0-1|1/2-1/2)\s*$', '', limpio)
    jugadas = [j for j in limpio.split(' ') if j.strip()]
    return jugadas

def load_evaluaciones(log_content):
    """Carga evaluaciones desde el contenido del log"""
    eval_values = re.findall(r'[\+\-]?\d+[,\.]\d{2}', log_content)
    return [float(value.replace(',', '.')) for value in eval_values]

def vectorizar_fen(fen, piezas):
    """Vectoriza una posición FEN"""
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
    """Extrae características de una posición FEN"""
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
    """Clasifica la calidad de una jugada"""
    if pd.isna(eval_actual) or pd.isna(eval_anterior):
        return 'desconocida'
    delta = eval_actual - eval_anterior
    if abs(delta) < 0.15:
        return 'buena'
    elif delta < -0.15:
        return 'error'
    else:
        return 'dudosa'

def procesar_datos(pgn_content, log_content, modelo_path):
    """Función principal de procesamiento"""
    # 1. Procesar PGN
    df = process_pgn(pgn_content)
    if df.empty:
        st.error("No se pudieron procesar las partidas del archivo PGN")
        return None
    
    movimientos = extraer_movimientos_desde_pgn(pgn_content)
    df['AN'] = [movimientos for _ in range(len(df))]
    df['SAN'] = df['AN'].apply(extraer_jugadas_san)
    df['ECO_Family'] = df['ECO'].str[0]
    df['Result'] = df['Result'].map({'1-0': 1, '1/2-1/2': 0, '0-1': -1})
    
    # 2. Evaluaciones
    evals = load_evaluaciones(log_content)
    df['evals'] = [evals for _ in range(len(df))]
    
    # 3. Preprocesamiento para cálculo posiciones FEN
    datos_fen = []
    piezas = {None: 0, chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3, chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6}
    
    progress_bar = st.progress(0)
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
        
        progress_bar.progress((idx + 1) / len(df))
    
    df_fens = pd.DataFrame(datos_fen)
    
    # Preprocesamiento de posiciones vectorizadas
    vector_fens = df_fens['FEN'].apply(lambda fen: vectorizar_fen(fen, piezas))
    fen_vector_df = pd.DataFrame(vector_fens.tolist(), columns=[f'FEN_{i}' for i in range(64)])
    df_fens = pd.concat([df_fens.reset_index(drop=True), fen_vector_df], axis=1)
    
    # 5. Preprocesamiento clasificación jugada
    df_fens['eval_anterior'] = df_fens.groupby('ID_partida')['eval'].shift(1)
    df_fens['calidad_jugada'] = df_fens.apply(lambda row: clasificar_jugada(row['eval'], row['eval_anterior']), axis=1)
    
    # Preprocesamiento de porcentaje de jugadas
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
    
    # Preprocesamiento de datos
    df_ml['eval_anterior'] = df_ml['eval_anterior'].fillna(0)
    df_ml['eval'] = df_ml['eval'].fillna(0)
    exclude_cols = [col for col in df_ml.columns if col.startswith('FEN_')]
    exclude_cols += ['BlackElo', 'ID_partida', 'jugada_num', 'SAN', 'White', 'Black', 'FEN', 'WhiteElo', 'Result']
    exclude_cols += ['material_blancas', 'material_negras', 'diferencia_material', 'turno', 'enroque_blancas', 'enroque_negras', 'jaque', 'mate']
    df_ml = df_ml.drop(exclude_cols, axis=1, errors='ignore')
    categorical_cols = ['ECO_Family', 'calidad_jugada']
    df_ml = pd.get_dummies(df_ml, columns=[col for col in categorical_cols if col in df_ml.columns], drop_first=True)
    
    # Carga del modelo
    modelo_rf = joblib.load(modelo_path)
    columnas_esperadas = modelo_rf.feature_names_in_
    for col in columnas_esperadas:
        if col not in df_ml.columns:
            df_ml[col] = 0
    df_ml = df_ml[columnas_esperadas]
    predicciones_elo = modelo_rf.predict(df_ml)
    df_ml['prediccion_elo_negro'] = predicciones_elo
    media_elo_negro = round(df_ml['prediccion_elo_negro'].mean())
    
    return media_elo_negro, df_ml, df_fens

# Interfaz de Streamlit
def main():
    st.title("♟️ Chess ELO Predictor")
    st.markdown("Sube un archivo PGN y un archivo LOG para predecir el ELO del jugador negro")
    
    # Configuración en columnas
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Archivo PGN")
        pgn_file = st.file_uploader("Sube tu archivo PGN", type=['pgn'], key="pgn")
        
    with col2:
        st.subheader("📊 Archivo LOG")
        log_file = st.file_uploader("Sube tu archivo LOG", type=['log', 'txt'], key="log")
    
    # Ruta del modelo (puedes cambiar esto según tu estructura)
    modelo_path = st.text_input("Ruta del modelo joblib", value="modelo.joblib")
    
    if st.button("🔍 Predecir ELO", type="primary"):
        if pgn_file is not None and log_file is not None:
            # Verificar que el modelo existe
            if not os.path.exists(modelo_path):
                st.error(f"No se encontró el modelo en la ruta: {modelo_path}")
                return
            
            with st.spinner("Procesando archivos..."):
                try:
                    # Leer contenido de los archivos
                    pgn_content = pgn_file.read().decode('utf-8')
                    log_content = log_file.read().decode('utf-8')
                    
                    # Procesar datos
                    resultado = procesar_datos(pgn_content, log_content, modelo_path)
                    
                    if resultado is not None:
                        media_elo_negro, df_ml, df_fens = resultado
                        
                        # Mostrar resultado principal
                        st.success(f"🎯 **ELO predicho del jugador negro: {media_elo_negro}**")
                        
                        # Mostrar estadísticas adicionales
                        st.subheader("📈 Estadísticas del análisis")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Jugadas analizadas", len(df_fens))
                        
                        with col2:
                            st.metric("Partidas procesadas", df_fens['ID_partida'].nunique())
                        
                        with col3:
                            st.metric("ELO promedio predicho", f"{media_elo_negro}")
                        
                        # Mostrar distribución de calidad de jugadas
                        if 'calidad_jugada' in df_fens.columns:
                            st.subheader("🎯 Distribución de calidad de jugadas")
                            calidad_counts = df_fens['calidad_jugada'].value_counts()
                            st.bar_chart(calidad_counts)
                        
                        # Mostrar muestra de datos
                        st.subheader("🔍 Muestra de datos procesados")
                        st.dataframe(df_fens.head(10))
                        
                        # Opción para descargar resultados
                        csv_data = df_ml.to_csv(index=False)
                        st.download_button(
                            label="📥 Descargar resultados completos (CSV)",
                            data=csv_data,
                            file_name="predicciones_elo.csv",
                            mime="text/csv"
                        )
                        
                except Exception as e:
                    st.error(f"Error al procesar los archivos: {str(e)}")
                    st.error("Por favor, verifica que los archivos tengan el formato correcto.")
        else:
            st.warning("Por favor, sube tanto el archivo PGN como el archivo LOG.")
    
    # Información adicional
    with st.expander("ℹ️ Información sobre el formato de archivos"):
        st.markdown("""
        **Archivo PGN:**
        - Debe contener partidas de ajedrez en formato PGN estándar
        - Incluir headers como Event, White, Black, Result, etc.
        - Los movimientos deben estar en notación algebraica estándar
        
        **Archivo LOG:**
        - Debe contener evaluaciones numéricas de las posiciones
        - Formato: números decimales con punto o coma como separador
        - Ejemplo: +0.23, -1.45, +2.10
        
        **Modelo:**
        - Archivo .joblib entrenado con scikit-learn
        - Debe ser compatible con las características generadas por este script
        """)

if __name__ == "__main__":
    main()