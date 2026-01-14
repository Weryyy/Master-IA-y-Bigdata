import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict

# ==========================================
# 1. DEFINICIÓN DEL ENTORNO UNO (Custom Gym)
# ==========================================


class UnoEnv(gym.Env):
    def __init__(self):
        super(UnoEnv, self).__init__()
        # Definimos el mazo: 4 colores x 10 números
        # Rojo, Azul, Verde, aZul(o) -> Z para diferenciar
        self.colores = ['R', 'A', 'V', 'Z']
        self.numeros = list(range(10))  # 0-9
        self.deck_base = [(c, n) for c in self.colores for n in self.numeros]

        # Acciones:
        # 0 a 39: Jugar carta específica (R0...R9, A0...A9, etc.)
        # 40: Robar carta (si no puedes jugar)
        self.action_space = spaces.Discrete(41)

        # El espacio de observación es complejo, así que lo devolveremos como tupla
        # y no lo definiremos estrictamente con gym.spaces para este ejemplo tabular.
        self.observation_space = spaces.Discrete(1)  # Placeholder

    def reset(self, seed=None):
        # Barajamos
        self.mazo = self.deck_base.copy()
        random.shuffle(self.mazo)

        # Repartimos 5 cartas a cada uno (Agente y Rival)
        self.mano_agente = [self.mazo.pop() for _ in range(5)]
        self.mano_rival = [self.mazo.pop() for _ in range(5)]

        # Carta inicial en la mesa
        self.mesa = self.mazo.pop()

        # Ordenamos mano para que el estado sea consistente
        self.mano_agente.sort(key=lambda x: (x[0], x[1]))

        return self._get_state(), {}

    def _get_state(self):
        # El estado es una tupla inmutable: (Carta_Mesa, Tu_Mano)
        # Convertimos la lista de mano a tupla para que pueda ser clave de diccionario
        return (self.mesa, tuple(self.mano_agente))

    def step(self, action):
        reward = 0
        done = False

        # --- TURNO DEL AGENTE ---
        if action == 40:  # Acción de ROBAR
            carta_robada = self.mazo.pop() if self.mazo else None
            if carta_robada:
                self.mano_agente.append(carta_robada)
            # Pequeña penalización por robar (queremos ganar rápido)
            reward = -0.1

        else:  # Intentar jugar una carta
            # Decodificar acción (ej: 0 -> R0)
            color_idx = action // 10
            num = action % 10
            if color_idx < 4:
                carta_a_jugar = (self.colores[color_idx], num)
            else:
                carta_a_jugar = None  # Acción inválida

            # Verificar si tiene la carta y si es legal
            if carta_a_jugar in self.mano_agente:
                if carta_a_jugar[0] == self.mesa[0] or carta_a_jugar[1] == self.mesa[1]:
                    # JUGADA VÁLIDA
                    self.mano_agente.remove(carta_a_jugar)
                    self.mesa = carta_a_jugar
                    reward = 1.0  # Recompensa por jugar bien

                    if len(self.mano_agente) == 0:
                        return self._get_state(), 100, True, False, {"msg": "GANASTE"}
                else:
                    # Carta incorrecta (no coincide color ni número)
                    reward = -10  # Castigo fuerte por intentar trampa
            else:
                # No tiene esa carta
                reward = -10  # Castigo fuerte por jugar carta que no tiene

        # Ordenar mano
        self.mano_agente.sort(key=lambda x: (x[0], x[1]))

        # --- TURNO DEL RIVAL (BOT SIMPLE) ---
        if not done and len(self.mano_agente) > 0:
            jugo = False
            # El rival no piensa, juega al azar lo que pueda
            random.shuffle(self.mano_rival)
            for carta in self.mano_rival:
                if carta[0] == self.mesa[0] or carta[1] == self.mesa[1]:
                    self.mano_rival.remove(carta)
                    self.mesa = carta
                    jugo = True
                    break

            if not jugo:
                if self.mazo:
                    self.mano_rival.append(self.mazo.pop())  # Roba

            if len(self.mano_rival) == 0:
                return self._get_state(), -50, True, False, {"msg": "PERDISTE"}

        # Si se acaba el mazo, empate/reset técnico (para no crashear)
        if len(self.mazo) == 0:
            done = True

        return self._get_state(), reward, done, False, {}

    def render(self):
        print(f"MESA: {self.mesa} | MANO: {self.mano_agente}")

# ==========================================
# 2. CONFIGURACIÓN DEL ENTRENAMIENTO
# ==========================================


env = UnoEnv()

# USAMOS UN DICCIONARIO EN LUGAR DE NP.ZEROS
# Esto permite "memoria infinita" (solo guarda los estados que visita)
q_table = defaultdict(lambda: np.zeros(env.action_space.n))

# Hiperparámetros
episodes = 50000    # Necesitamos muchos episodios
alpha = 0.1         # Learning rate
gamma = 0.9         # Discount factor
epsilon = 1.0       # Exploración
epsilon_decay = 0.99995
min_epsilon = 0.05

outcomes = []

# Mapeo inverso para saber qué carta es cada acción
idx_to_card = {}
counter = 0
colores = ['R', 'A', 'V', 'Z']
for c in colores:
    for n in range(10):
        idx_to_card[counter] = f"{c}{n}"
        counter += 1
idx_to_card[40] = "ROBAR"

# ==========================================
# 3. BUCLE DE ENTRENAMIENTO
# ==========================================
print("Iniciando entrenamiento de UNO... (Esto puede tardar unos segundos)")

for i in range(episodes):
    state, _ = env.reset()
    done = False

    # Si el episodio tarda demasiado (bucle infinito de robar), lo cortamos
    pasos = 0

    while not done and pasos < 100:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explorar
        else:
            # Explotar: buscamos en el diccionario
            # Si el estado es nuevo, devuelve un array de ceros automáticamente
            action = np.argmax(q_table[state])

        next_state, reward, done, truncated, info = env.step(action)

        # Q-Learning update
        old_value = q_table[state][action]
        next_max = np.max(q_table[next_state])

        new_value = old_value + alpha * (reward + gamma * next_max - old_value)
        q_table[state][action] = new_value

        state = next_state
        pasos += 1

    if epsilon > min_epsilon:
        epsilon *= epsilon_decay

    if (i+1) % 5000 == 0:
        print(f"Episodio {i+1}/{episodes} completado. Epsilon: {epsilon:.4f}")

print("Entrenamiento finalizado.")
print(f"Número de estados únicos descubiertos: {len(q_table)}")


# ==========================================
# 4. DEMOSTRACIÓN (JUGAR PARTIDA)
# ==========================================
print("\n==============================")
print("     PARTIDA DE DEMOSTRACIÓN")
print("==============================")

state, _ = env.reset()
done = False
input("Presiona ENTER para ver al agente jugar...")

step_count = 0
while not done and step_count < 20:
    env.render()

    # Agente elige acción
    # Imprimimos los valores Q para ver qué está pensando
    q_values = q_table[state]
    best_action = np.argmax(q_values)

    accion_nombre = idx_to_card.get(best_action, "DESC")
    print(
        f"Agente elige: {accion_nombre} (Valor Q: {q_values[best_action]:.2f})")

    # Validar visualmente si intenta hacer trampa
    if q_values[best_action] < 0:
        print("⚠️ El agente está confundido o acorralado (valores negativos).")

    state, reward, done, _, info = env.step(best_action)
    print("------------------------------")
    step_count += 1

    if "msg" in info:
        print(f"RESULTADO: {info['msg']}")

if step_count >= 20:
    print("La partida se alargó demasiado (empate técnico).")
