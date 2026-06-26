import numpy as np
import gymnasium as gym

class RLModel:
    def __init__(self):
        # Tasa de aprendizaje: qué tan rápido actualiza sus conocimientos
        self.learning_rate = 0.8
        # Factor de descuento: cuánto valora las recompensas futuras (0=solo presente, 1=muy futuro)
        self.gamma = 0.95
        # Epsilon: probabilidad de explorar (acción aleatoria) vs explotar (mejor acción conocida)
        self.epsilon = 1.0
        # Cuánto reduce epsilon por episodio (va explorando menos con el tiempo)
        self.epsilon_decay = 0.999
        # Epsilon mínimo: siempre mantiene un pequeño % de exploración
        self.epsilon_min = 0.01
        # Cantidad de partidas que va a jugar para aprender
        self.episodes = 10000

        # Crea el entorno y guarda cuántos estados y acciones tiene
        self.env, self.n_states, self.n_actions = self.setup_env()

        # Tabla Q: matriz de (estados x acciones) que almacena el "valor" de cada acción en cada estado
        # Empieza en cero porque no sabe nada todavía
        self.q_table = np.zeros((self.n_states, self.n_actions))

    def setup_env(self, is_slippery=False):
        # Crea el entorno FrozenLake: un lago helado de 4x4 donde hay que llegar a la meta sin caer
        # is_slippery=False hace que el agente se mueva exactamente a donde quiere (más fácil de aprender)
        self.env = gym.make(
            'FrozenLake-v1',
            desc=None,
            map_name='4x4',
            is_slippery=is_slippery
        )
        return self.env, self.env.observation_space.n, self.env.action_space.n

    def choose_action(self, state):
        # Genera un número aleatorio entre 0 y 1
        random_n = np.random.random()

        # Si el número es menor que epsilon, explora (elige una acción al azar)
        if random_n < self.epsilon:
            return self.env.action_space.sample()

        # Si no, explota: elige la acción con mayor valor en la tabla Q para este estado
        return np.argmax(self.q_table[state])

    def update_qtable(self, state, action, reward, next_state, done):
        # Calcula el valor objetivo: recompensa actual + valor futuro descontado
        target = reward + self.gamma * np.max(self.q_table[next_state])

        # Si el episodio terminó, no hay recompensa futura
        if done:
            target = reward

        # Actualiza la tabla Q usando la fórmula de Q-learning
        # La celda se acerca al target según la tasa de aprendizaje
        self.q_table[state][action] = self.q_table[state][action] + self.learning_rate * (target - self.q_table[state][action])

    def train(self):
        # Contador de victorias
        count = 0

        for episode in range(self.episodes):
            # Reinicia el entorno al inicio de cada episodio
            state, info = self.env.reset()
            done = False

            while not done:
                # Elige una acción según la política actual
                action = self.choose_action(state)
                # Ejecuta la acción y obtiene el resultado
                next_state, reward, terminated, truncated, info = self.env.step(action)

                # Si llegó a la meta, suma una victoria
                if reward == 1.0:
                    count += 1

                # Actualiza la tabla Q con lo que aprendió
                self.update_qtable(state, action, reward, next_state, terminated)
                state = next_state
                done = terminated or truncated

            # Reduce epsilon: con el tiempo, explora menos y aprovecha más lo aprendido
            self.epsilon = self.epsilon * self.epsilon_decay

            # Nunca baja del mínimo establecido
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

            # Muestra el progreso cada 200 episodios
            if episode % 200 == 0:
                print(f'Episodio {episode} | Tasa de victorias: {(count / (episode + 1)) * 100:.1f}%')

        # Al terminar el entrenamiento, guarda la tabla Q en un archivo
        self.save_qtable()

    def save_qtable(self, filename='q_table.npy'):
        # Guarda la tabla Q en un archivo .npy para no tener que reentrenar
        np.save(filename, self.q_table)
        print(f'Tabla Q guardada en "{filename}"')

    def load_qtable(self, filename='q_table.npy'):
        # Carga una tabla Q previamente entrenada desde un archivo
        self.q_table = np.load(filename)
        print(f'Tabla Q cargada desde "{filename}"')

    def evaluate(self, episodes=100, filename='q_table.npy'):
        # Carga la tabla Q entrenada
        self.load_qtable(filename)

        wins = 0
        # Desactiva la exploración: el agente solo usa lo que aprendió
        epsilon_backup = self.epsilon
        self.epsilon = 0.0

        for _ in range(episodes):
            state, info = self.env.reset()
            done = False

            while not done:
                # Solo elige la mejor acción conocida, sin aleatoriedad
                action = np.argmax(self.q_table[state])
                next_state, reward, terminated, truncated, info = self.env.step(action)
                state = next_state
                done = terminated or truncated

                if reward == 1.0:
                    wins += 1

        # Restaura epsilon por si se vuelve a entrenar
        self.epsilon = epsilon_backup

        win_rate = (wins / episodes) * 100
        print(f'Evaluación sobre {episodes} episodios | Tasa de victorias: {win_rate:.1f}%')
        return win_rate


if __name__ == '__main__':
    rl_model = RLModel()

    # Entrena el modelo (guarda la tabla Q automáticamente al terminar)
    rl_model.train()

    # Evalúa el modelo entrenado usando la tabla Q guardada
    rl_model.evaluate(episodes=100)
