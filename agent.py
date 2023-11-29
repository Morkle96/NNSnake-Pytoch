from replay_buffer import ReplayBufferNumpy
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Agent():
    def __init__(self, board_size=10, frames=2, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True,
                 version=''):

        self._board_size = board_size
        self._n_frames = frames
        self._buffer_size = buffer_size
        self._n_actions = n_actions
        self._gamma = gamma
        self._use_target_net = use_target_net
        self._input_shape = (self._board_size, self._board_size, self._n_frames)
        # reset buffer also initializes the buffer
        self.reset_buffer()
        self._board_grid = np.arange(0, self._board_size**2)\
                             .reshape(self._board_size, -1)
        self._version = version

    def get_gamma(self):
        return self._gamma

    def reset_buffer(self, buffer_size=None):
        if(buffer_size is not None):
            self._buffer_size = buffer_size
        self._buffer = ReplayBufferNumpy(self._buffer_size, self._board_size, 
                                    self._n_frames, self._n_actions)

    def get_buffer_size(self):
        return self._buffer.get_current_size()

    def add_to_buffer(self, board, action, reward, next_board, done, legal_moves):
        self._buffer.add_to_buffer(board, action, reward, next_board, 
                                   done, legal_moves)

    def save_buffer(self, file_path='', iteration=None):
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), 'wb') as f:
            pickle.dump(self._buffer, f)

    def load_buffer(self, file_path='', iteration=None):
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), 'rb') as f:
            self._buffer = pickle.load(f)

    def _point_to_row_col(self, point):
        return (point//self._board_size, point%self._board_size)

    def _row_col_to_point(self, row, col):
        return row*self._board_size + col

class DeepQLearningAgent(Agent):
    def __init__(self, board_size=10, frames=4, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True,
                 version=''):
        
        #Using GPU or CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("Running on GPU")
        else:
            self.device = torch.device("cpu")
            print("Running on CPU")

        Agent.__init__(self, board_size=board_size, frames=frames, buffer_size=buffer_size,
                 gamma=gamma, n_actions=n_actions, use_target_net=use_target_net,
                 version=version)
        self.reset_models()

    def reset_models(self):
        self._model = self._agent_model().to(self.device)
        if self._use_target_net:
            self._target_net = self._agent_model().to(self.device)
            self.update_target_net()

    def _prepare_input(self, board):
        if isinstance(board, np.ndarray):
            board = torch.tensor(board, dtype=torch.float32).to(self.device)
        
        #Change NCHW to NHWC
        if board.ndim == 3:
            board = board.permute(2, 0, 1).unsqueeze(0)
        else:
            board = board.permute(0, 3, 1, 2)
        
        #Normalize board
        board = self._normalize_board(board.clone())
        return board.to(self.device)

    def _get_model_outputs(self, board, model=None):
        #Prepare input
        board = self._prepare_input(board)
        if model is None:
            model = self._model
        model_outputs = model(board).to(self.device)
        return model_outputs

    def _normalize_board(self, board):
        return board / 4.0

    def move(self, board, legal_moves, value=None):
        model_outputs = self._get_model_outputs(board).detach().cpu().numpy()
        return np.argmax(np.where(legal_moves==1, model_outputs, -np.inf), axis=1)

    def _agent_model(self):
        #Loading the model
        model = CNN_DeepQLearningAgent()
        return model

    def set_weights_trainable(self):
        for layer in self._model.layers:
            layer.trainable = False
        for s in ['action_prev_dense', 'action_values']:
            self._model.get_layer(s).trainable = True
        self._model.compile(optimizer = self._model.optimizer, 
                            loss = self._model.loss)

    def get_action_proba(self, board, values=None):
        model_outputs = self._get_model_outputs(board, self._model)
        model_outputs = np.clip(model_outputs, -10, 10)
        model_outputs = model_outputs - model_outputs.max(axis=1).reshape((-1,1))
        model_outputs = np.exp(model_outputs)
        model_outputs = model_outputs/model_outputs.sum(axis=1).reshape((-1,1))
        return model_outputs

    def save_model(self, file_path='', iteration=None):
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
            
        torch.save(self._model.state_dict(), "{}/model_{:04d}.pt".format(file_path, iteration))
        if(self._use_target_net):
            torch.save(self._target_net.state_dict(), "{}/model_{:04d}_target.pt".format(file_path, iteration))

    def load_model(self, file_path='', iteration=None):
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0

        self._model.load_state_dict(torch.load("{}/model_{:04d}.pt".format(file_path, iteration)))
        if(self._use_target_net):
            self._target_net.load_state_dict(torch.load("{}/model_{:04d}_target.pt".format(file_path, iteration)))

    def print_models(self):
        print('Training Model')
        print(self._model)
        if(self._use_target_net):
            print('Target Network')
            print(self._target_net)

    def train_agent(self, batch_size=32, num_games=1, reward_clip=False):
        learning_rate = 0.0005
        #Setting the optimizer using the RMSprop
        optimizer = optim.RMSprop(self._model.parameters(), learning_rate)

        s, a, r, next_s, done, legal_moves = self._buffer.sample(batch_size)

        if reward_clip:
            r = np.sign(r)

        s = torch.tensor(s, dtype=torch.float32).to(self.device)
        a = torch.tensor(a, dtype=torch.float32).to(self.device)
        r = torch.tensor(r, dtype=torch.float32).to(self.device)
        next_s = torch.tensor(next_s, dtype=torch.float32).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).to(self.device)
        legal_moves = torch.tensor(legal_moves, dtype=torch.float32).to(self.device)

        self._model.train()

        current_model = self._target_net if self._use_target_net else self._model

        with torch.no_grad():
            next_model_outputs = self._get_model_outputs(next_s, current_model)

        inff = torch.tensor(-1000000).to(self.device) #-1000000 instead of -inf
        discounted_reward = r + (self._gamma * torch.max(torch.where(legal_moves == 1, next_model_outputs, inff),
                                                         dim=1).values.reshape(-1, 1)) * (1 - done)

        target = self._get_model_outputs(s, current_model)
        target = (1 - a) * target + a * discounted_reward

        optimizer.zero_grad()

        #Using Huberloss for loss function
        criterion = nn.HuberLoss()
        #Calculating the loss
        loss = criterion(self._get_model_outputs(s), target)

        loss.backward()
        optimizer.step()
        
        #Returns the value in the loss function
        return loss.item()

    def update_target_net(self):
        if(self._use_target_net):
            self._target_net.load_state_dict(self._model.state_dict())

    def compare_weights(self):
        for name, param in self.model.named_parameters():
            target_param = getattr(self.target_net, name)
            if torch.equal(param.data, target_param.data):
                print(f'Layer {name} Weights Match')
            else:
                print(f'Layer {name} Weights Do Not Match')

    def copy_weights_from_agent(self, agent_for_copy):
        assert isinstance(agent_for_copy, self), "Agent type is required for copy"
        self.model.load_state_dict(agent_for_copy.model.state_dict())
        if self.use_target_net:
            self.target_net.load_state_dict(agent_for_copy.model.state_dict())

class CNN_DeepQLearningAgent(nn.Module):
    def __init__(self, input_channels=2, num_actions=4):
        super(CNN_DeepQLearningAgent, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, (3,3))
        self.conv2 = nn.Conv2d(16, 32, (3,3))
        self.conv3 = nn.Conv2d(32, 64, (5,5))
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(256, 64)
        self.out = nn.Linear(64, num_actions)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flat(x)
        x = F.relu(self.fc1(x))
        x = self.out(x)
        
        return x