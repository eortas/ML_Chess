import chess.pgn
import chess.engine
import pandas as pd
from datetime import datetime

# Función para leer el PGN y crear el DataFrame
def process_pgn(pgn_file):
    # Cargar el archivo PGN
    pgn = open(pgn_file)
    game = chess.pgn.read_game(pgn)

    # Inicia el motor de Stockfish de manera sincrónica
    engine = chess.engine.SimpleEngine.popen_uci("/stockfish/stockfish.exe")  # Asegúrate de poner la ruta correcta a Stockfish

    # Extraer información del encabezado del PGN
    event = game.headers["Event"]
    white_player = game.headers["White"]
    black_player = game.headers["Black"]
    result = game.headers["Result"]
    utc_date = game.headers["UTCDate"]
    utc_time = game.headers["UTCTime"]
    white_elo = game.headers["WhiteElo"]
    black_elo = game.headers["BlackElo"]
    white_rating_diff = game.headers["WhiteRatingDiff"]
    black_rating_diff = game.headers["BlackRatingDiff"]
    eco = game.headers["ECO"]
    opening = game.headers["Opening"]
    time_control = game.headers["TimeControl"]
    termination = game.headers["Termination"]
    an = game.board().variation_san(game.mainline_moves())  # Notación algebraica completa

    # Lista para almacenar las evaluaciones de los movimientos
    evals = []

    # Crear el tablero en la posición inicial
    board = game.board()

    # Iterar sobre los movimientos del juego y calcular las evaluaciones
    for i, movimiento in enumerate(game.mainline_moves()):
        try:
            board.push(movimiento)

            # Evaluar el movimiento con Stockfish
            result = engine.analyse(board, chess.engine.Limit(time=1.0))  # Aumentamos el tiempo de análisis
            evaluation = result["score"].relative.score(mate_score=10000)  # Evaluación en centésimas de peón

            # Normalizar las evaluaciones, dividiendo entre 100 si es un valor entero
            normalized_eval = evaluation / 100.0 if isinstance(evaluation, int) else evaluation

            evals.append(normalized_eval)  # Guardar la evaluación

        except chess.engine.EngineError as e:
            print(f"Error con el motor de ajedrez en el movimiento {movimiento}: {e}")
            evals.append(None)  # Si hay un error, agregar None

    # Cerrar el motor
    engine.quit()

    # Crear el DataFrame con la información por partida
    df = pd.DataFrame({
        "Event": [event],
        "White": [white_player],
        "Black": [black_player],
        "Result": [result],
        "UTCDate": [utc_date],
        "UTCTime": [utc_time],
        "WhiteElo": [white_elo],
        "BlackElo": [black_elo],
        "WhiteRatingDiff": [white_rating_diff],
        "BlackRatingDiff": [black_rating_diff],
        "ECO": [eco],
        "Opening": [opening],
        "TimeControl": [time_control],
        "Termination": [termination],
        "AN": [an],
        "evals": [evals]
    })

    return df

# Llamar a la función con el archivo PGN y guardar el DataFrame
df_result = process_pgn("predict.pgn")

# Guardar el DataFrame en un archivo CSV
df_result.to_csv("evaluaciones_partida.csv", index=False)

# Mostrar el DataFrame
print(df_result)
