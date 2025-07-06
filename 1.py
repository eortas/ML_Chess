import chess.pgn
import chess.engine
import pandas as pd

# Abre el archivo PGN
pgn = open("predict.pgn")
game = chess.pgn.read_game(pgn)

# Inicia el motor de Stockfish
engine = chess.engine.SimpleEngine.popen_uci("/stockfish/stockfish.exe")  # Asegúrate de poner la ruta correcta a Stockfish

# Extraer información del encabezado del PGN
white_player = game.headers["White"]
black_player = game.headers["Black"]
result = game.headers["Result"]
white_elo = game.headers["WhiteElo"]
black_elo = game.headers["BlackElo"]
eco = game.headers["ECO"]
time_control = game.headers["TimeControl"]
termination = game.headers["Termination"]
eco_family = eco[0]  # Primera letra del ECO (familia)
an = game.board().variation_san(game.mainline_moves())  # Notación algebraica completa

# Lista para almacenar las evaluaciones de los movimientos
evals = []

# Crear el tablero en la posición inicial
board = game.board()

# Iterar sobre los movimientos del juego
for i, movimiento in enumerate(game.mainline_moves()):
    try:
        # Empujar el movimiento al tablero
        board.push(movimiento)

        # Evaluar el movimiento con Stockfish
        result = engine.analyse(board, chess.engine.Limit(time=1.0))  # Aumentamos el tiempo de análisis
        evaluation = result["score"].relative.score(mate_score=10000)  # Evaluación en centésimas de peón

        evals.append(evaluation)  # Guardar la evaluación

    except chess.engine.EngineError as e:
        print(f"Error con el motor de ajedrez en el movimiento {movimiento}: {e}")
        evals.append(None)

# Cerrar el motor
engine.quit()

# Crear el DataFrame con la información
df = pd.DataFrame({
    "White": [white_player],
    "Black": [black_player],
    "Result": [result],
    "WhiteElo": [white_elo],
    "BlackElo": [black_elo],
    "ECO": [eco],
    "TimeControl": [time_control],
    "Termination": [termination],
    "AN": [an],
    "ECO_Family": [eco_family],
    "evals": [evals]
})

# Dividir la columna 'evals' entre 100
df['evals'] = df['evals'].apply(lambda x: [i/100 for i in x] if isinstance(x, list) else x)

# Si quieres guardar el DataFrame en un archivo CSV
df.to_csv("evaluaciones_partida.csv", index=False)
