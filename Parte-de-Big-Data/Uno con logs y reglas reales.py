import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import defaultdict
import time


class UnoReglasReales(gym.Env):
    def __init__(self):
        super(UnoReglasReales, self).__init__()
        self.colores = ['Rojo', 'Azul', 'Verde']
        self.numeros = [1, 2, 3]
        self.deck_base = [(c, n) for c in self.colores for n in self.numeros]
        # Acciones: 0: Tirar más baja legal, 1: Tirar más alta legal, 2: Robar
        self.action_space = spaces.Discrete(3)

    def _get_state(self):
        color_m, num_m = self.mesa
        # REGLA CORREGIDA: Coincide Color O Coincide Número
        legales = [c for c in self.mano_agente if c[0]
                   == color_m or c[1] == num_m]

        t1 = any(c[1] == 1 for c in legales)
        t2 = any(c[1] == 2 for c in legales)
        t3 = any(c[1] == 3 for c in legales)
        return (color_m, num_m, t1, t2, t3, len(self.mano_agente))

    def step(self, action):
        reward = 0
        done = False
        color_m, num_m = self.mesa
        # Buscamos cartas legales con la nueva restricción
        legales = [c for c in self.mano_agente if c[0]
                   == color_m or c[1] == num_m]

        if action == 0 or action == 1:  # INTENTAR JUGAR
            if legales:
                legales.sort(key=lambda x: x[1], reverse=(action == 1))
                carta = legales[0]
                self.mano_agente.remove(carta)
                self.mesa = carta
                reward = 15  # Recompensa por jugada legal
            else:
                reward = -20  # Castigo por intentar jugar sin cartas válidas
                # Si falla, no hace nada (se queda en el mismo estado para que aprenda el error)

        elif action == 2:  # ROBAR
            # Solo penalizamos si el agente PODÍA jugar y decidió robar
            if legales:
                reward = -10  # "Malas prácticas": Robar teniendo jugada
            else:
                reward = -1  # Necesidad: No tenía otra opción

            if self.mazo:
                self.mano_agente.append(self.mazo.pop())

        # Check Victoria
        if len(self.mano_agente) == 0:
            return self._get_state(), 100, True, False, {"msg": "GANASTE EL JUEGO"}

        # Turno Rival (aplicando las mismas reglas)
        legales_rival = [c for c in self.mano_rival if c[0]
                         == self.mesa[0] or c[1] == self.mesa[1]]
        if legales_rival:
            c = random.choice(legales_rival)
            self.mano_rival.remove(c)
            self.mesa = c
        elif self.mazo:
            self.mano_rival.append(self.mazo.pop())

        if len(self.mano_rival) == 0:
            return self._get_state(), -50, True, False, {"msg": "PERDISTE CONTRA EL RIVAL"}

        return self._get_state(), reward, False, False, {}

    def reset(self, seed=None):
        self.mazo = self.deck_base.copy() * 4
        random.shuffle(self.mazo)
        self.mano_agente = [self.mazo.pop() for _ in range(5)]
        self.mano_rival = [self.mazo.pop() for _ in range(5)]
        self.mesa = self.mazo.pop()
        return self._get_state(), {}


# --- Entrenamiento (Subimos episodios para las nuevas reglas) ---
env = UnoReglasReales()
q_table = defaultdict(lambda: np.zeros(env.action_space.n))
print("Entrenando con reglas reales (Mismo Color o Mismo Número)...")
for i in range(60000):
    state, _ = env.reset()
    done = False
    while not done:
        # Epsilon-greedy
        action = env.action_space.sample() if random.random(
        ) < 0.1 else np.argmax(q_table[state])
        next_state, reward, done, _, _ = env.step(action)
        # Update Q-Value
        q_table[state][action] += 0.1 * \
            (reward + 0.95 *
             np.max(q_table[next_state]) - q_table[state][action])
        state = next_state

# --- Demostración Visual ---
print("\n" + "!"*40)
print("DEMOSTRACIÓN: REGLAS DE COINCIDENCIA")
print("!"*40)

state, _ = env.reset()
done = False
while not done:
    c_m, n_m = env.mesa
    legales = [c for c in env.mano_agente if c[0] == c_m or c[1] == n_m]

    print(f"\n[MESA]: {c_m} {n_m}")
    print(f"[TU MANO]: {env.mano_agente}")
    print(f"[CARTAS LEGALES]: {legales if legales else 'NINGUNA'}")

    action = np.argmax(q_table[state])
    action_name = ["JUGAR BAJA", "JUGAR ALTA", "ROBAR"][action]

    # Verificamos qué pasaría
    next_state, reward, done, _, info = env.step(action)

    print(f"[IA DECIDE]: {action_name}")
    print(f"[RESULTADO]: {reward} puntos")

    if action != 2 and not legales:
        print("❌ ERROR: La IA intentó jugar pero no tenía cartas del mismo color o número.")
    elif action == 2 and legales:
        print("⚠️ COBARDÍA: La IA robó teniendo cartas para jugar.")
    elif reward > 0:
        print("✅ MOVIMIENTO LEGAL: Coincidencia de color o número detectada.")

    state = next_state
    if "msg" in info:
        print(f"\n*** {info['msg']} ***")
    time.sleep(1.5)
