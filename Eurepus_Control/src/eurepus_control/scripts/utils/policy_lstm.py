import torch
from torch import nn
import onnx
import onnxruntime as ort


class EurepusPolicyLSTM(nn.Module):
    def __init__(
        self, 
        input_size,
        hidden_sizes,
        output_size,
        pth_file: str
    ) -> None:
        super().__init__()

        self._pth = torch.load(pth_file, map_location=torch.device('cpu'))
        self._weights = self._pth["model"]
        self._num_mlp_layers = 2

        # MLP linear layers
        layers = []
        for i in range(self._num_mlp_layers):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_sizes[i]))
            else:
                layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(nn.ELU())

        # LSTM layer
        self.lstm = nn.LSTM(hidden_sizes[-1], hidden_sizes[-1], 1, batch_first=True)

        self.mlp_layers = nn.Sequential(*layers)

        j = 0
        for layer in self.mlp_layers:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.data = self._weights["a2c_network.actor_mlp.{}.weight".format(j)]
                layer.bias.data = self._weights["a2c_network.actor_mlp.{}.bias".format(j)]
                j+=2

        
        self.lstm.weight_ih_l0.data = self._weights["a2c_network.rnn.rnn.weight_ih_l0"]
        self.lstm.weight_hh_l0.data = self._weights["a2c_network.rnn.rnn.weight_hh_l0"]
        self.lstm.bias_ih_l0.data = self._weights["a2c_network.rnn.rnn.bias_ih_l0"]
        self.lstm.bias_hh_l0.data = self._weights["a2c_network.rnn.rnn.bias_hh_l0"]

        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

        self.output_layer.weight.data = self._weights["a2c_network.mu.weight"]
        self.output_layer.bias.data = self._weights["a2c_network.mu.bias"]

        self.register_buffer("_obs_mean", self._weights["running_mean_std.running_mean"].unsqueeze(0))
        self.register_buffer("_obs_std", self._weights["running_mean_std.running_var"].sqrt().unsqueeze(0))

        # self._model = nn.Sequential(self.hidden_layers, self.output_layer)
        
        self.layers = nn.ModuleList([self.mlp_layers, self.lstm, self.output_layer])

        self.eval()

    def forward(self, obs, h0, c0):
        norm_obs = ((obs - self._obs_mean) / self._obs_std).to(torch.float32)
        norm_obs = norm_obs
        x = self.mlp_layers(norm_obs)
        x, (hn, cn) = self.lstm(x, (h0, c0))
        action = self.output_layer(x)
        action = torch.clamp(action, -1.0, 1.0)
        return action, hn, cn
    

if __name__ == '__main__':
    # Test the policy LSTM model
    policy = EurepusPolicyLSTM(31, [128, 64, 64], 12, '../../models/eurepus_066.pth')
    policy.eval()

    obs = torch.randn(1,31)
    (h0, c0) = (torch.randn(1,64), torch.randn(1,64))
    action_torch, hn_torch, cn_torch = policy(obs, h0, c0)

    torch.onnx.export(policy,                  # model being run
                  (torch.randn(1,31), torch.randn(1,64), torch.randn(1,64)),        # model input (or a tuple for multiple inputs)
                  "eurepus_policy.onnx",# where to save the model (can be a file or file-like object)
                  input_names = ['input', 'h0', 'c0'],              # the model's input names
                  output_names = ['output', 'hn', 'cn'],            # the model's output namesd
                  export_params=True)            

    onnx_model = onnx.load("eurepus_policy.onnx")
    onnx.checker.check_model(onnx_model)
    ort_sess = ort.InferenceSession("eurepus_policy.onnx")

    ort_inputs = {'input': obs.numpy(), 'h0': h0.numpy(), 'c0': c0.numpy()}
    action_onnx, hn_onnx, cn_onnx = ort_sess.run(None, ort_inputs)

    # Assert the output of the PyTorch model and the ONNX model
    assert torch.allclose(action_torch, torch.tensor(action_onnx), atol=1e-5)
    assert torch.allclose(hn_torch, torch.tensor(hn_onnx), atol=1e-5)
    assert torch.allclose(cn_torch, torch.tensor(cn_onnx), atol=1e-5)
    



