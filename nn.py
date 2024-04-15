import numpy as np

class NeuralNetwork:
    def __init__(self, input_dim=False, hidden_dim=False, activation="tanh"):
        if input_dim and hidden_dim:
            if activation not in ["tanh", "relu", "softmax", "sigmoid", None]:
                raise ValueError("hãy nhập một hàm kích hoạt đúng")
            self.activation = activation
            self.weights = np.random.randn(input_dim, hidden_dim)
            self.biases = np.random.randn(hidden_dim)

    def softmax(self, x):
        try:
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        except:
            exp_x = np.exp(x - np.max(x))
            output = exp_x / np.sum(exp_x)
        return output
    
    def tanh(self, x):
        return np.tanh(x)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, inputs):
        output = np.dot(inputs, self.weights) + self.biases
        if self.activation == "tanh":
            return self.tanh(output)
        if self.activation == "relu":
            return self.relu(output)
        if self.activation == "sigmoid":
            return self.sigmoid(output)
        if self.activation == "softmax":
            return self.softmax(output)
        else:
            return output
        
    def gradient_descent(self, x, y, y_pred):
        n = float(len(x))
        update_weights = -np.dot(x.T, y - y_pred) / n
        update_biases = -np.mean(y - y_pred, axis=0)
        return update_weights, update_biases

    def compute_cost(self, y, y_pred):
            return np.mean(np.square(y - y_pred))
    
    def backward(self, layers, x, y, epochs=2, lr=0.001, limit_cost_decimal=3, limit_grad_finding=100000):
        assert isinstance(layers, list), "cần 1 danh sách chứa các layers"
        pre_loss = 1e100
        break_all = False
        for i in range(epochs):
            for f in range(limit_grad_finding):
                layers_op_final = x
                for layer in layers:
                    layer_output = layer.forward(layers_op_final)
                    update_weights, update_bias = self.gradient_descent(y=y, x=layers_op_final, y_pred=layer_output)
                    layer.weights -= lr * update_weights
                    layer.biases -= lr * update_bias
                    layers_op_final = layer_output

                loss = self.compute_cost(y_pred=layers_op_final, y=y)

                if f+10 >= limit_grad_finding:
                    break_all = True

                if round(loss, limit_cost_decimal) < round(pre_loss, limit_cost_decimal):
                    pre_loss = loss
                    break

            if break_all == True:
                print("vượt quá giới hạn tìm độ dốc")
                break

            print(f"Epoch {i+1}, Loss: {pre_loss}")
    
    def predict(self, x, layers):
        layers_op_final = x
        for layer in layers:
            layers_op_final = layer.forward(layers_op_final)
        return layers_op_final


# example inputs
x = np.array([
              [-1.2363188801691263, 3.0841715827615026, 12.03387711923264, -3.3586880359025404], 
              [-4.663450296841878, 12.979621321604862, -9.72758766810308, 9.17412767057377], 
              [-7.117883096789941, 9.805993492085843, 3.595913414267891, -0.3916747814772092], 
              [0.262632533772451, -4.6621570838899915, -2.6180871559568883, 3.575197991828804], 
              [16.028124156440498, -6.315977482458913, 2.4777774443417075, -4.1628653375100715],
              [-1.645301691263, 3.0841715827615026, 12.03387711923264, -3.3586880359025404], 
              [-4.663450296841878, 12.979624636634862, -9.72745546810308, 9.17412767057377], 
              [-7.117883096789941, 9.805993492085843, 3.5954232267891, -0.391875814772092], 
              [0.262632533772451, -4.6621570838899915, -2.61808355368883, 3.575197991828804], 
              [16.028124156440498, -6.31524358913, 2.4777774443417075, -4.1628653375100715],
              [-1.645301691263, 3.0841715827615026, 12.03387711923264, -3.3586880359025404], 
              [-4.663450296841878, 12.979624636634862, -9.72745546810308, 9.17412767057377], 
              [-7.117883096789941, 9.805993492085843, 3.5954232267891, -0.391875814772092], 
              [0.262632533772451, -4.6621570838899915, -2.61808355368883, 3.575197991828804], 
              [16.028124156440498, -6.31524358913, 2.4777774443417075, -4.1628653375100715]
])

y = np.array([
    [0],
    [1],
    [2],
    [3],
    [4],
    [0],
    [1],
    [2],
    [3],
    [4], 
    [0],
    [1],
    [2],
    [3],
    [4]])

# mô hình phân loại đa lớp
layers = [
NeuralNetwork(input_dim=4, hidden_dim=1000, activation="tanh"),
NeuralNetwork(input_dim=1000, hidden_dim=1000, activation="tanh"),
NeuralNetwork(input_dim=1000, hidden_dim=1000, activation="tanh"),
NeuralNetwork(input_dim=1000, hidden_dim=1000, activation="tanh"),
NeuralNetwork(input_dim=1000, hidden_dim=1000, activation="tanh"),
NeuralNetwork(input_dim=1000, hidden_dim=4, activation="softmax")
]

model = NeuralNetwork()
model.backward(layers=layers, x=x, y=y, epochs=500, lr=0.001, limit_cost_decimal=4,
                limit_grad_finding=200)

# thử nghiệm mô hình
while True:
    i = int(input("Bạn: "))
    y_pred = model.predict(layers=layers, x=np.array([x[i]]))
    print(f"y test: {y[i][0]}")
    print("y predict: ", np.argmax(y_pred))
    print()
