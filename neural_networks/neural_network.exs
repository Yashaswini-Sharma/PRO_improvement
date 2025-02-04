import IEx.Helpers

Mix.install([
  {:nx, "~> 0.5"}
])

defmodule NeuralNetwork do
  def trying() do
    n = {2, 3, 3, 1}
    key = Nx.Random.key(1)
    {w1, key} = Nx.Random.normal(key, 0, 0.01, shape: {elem(n, 1), elem(n, 0)})
    {w2, key} = Nx.Random.normal(key, 0, 0.01, shape: {elem(n, 2), elem(n, 1)})
    {w3, key} = Nx.Random.normal(key, 0, 0.01, shape: {elem(n, 3), elem(n, 2)})
    {b1, key} = Nx.Random.normal(key, 0, 1, shape: {elem(n, 1), 1})

    {b2, key} = Nx.Random.normal(key, 0, 1, shape: {elem(n, 2), 1})
    {b3, key} = Nx.Random.normal(key, 0, 1, shape: {elem(n, 3), 1})

    {a0, y, m} = prepare_data(n)
    {y_hat, a1, a2} = feed_forward( w1, w2, w3, b1, b2, b3, a0)

    epochs = 1000
    for i <- 0..epochs do
      {w3, w2, w1, b3, b1, b2} = train(a0, y, m, a2, a1, w1, w2, w3, b1, b2, b3, i)
    end
    costs = []
  end
  def train(a0, y, m, a2, a1, w1, w2, w3, b1, b2, b3, i) do
    alpha = 0.1
    {y_hat, _a1, _a2} = feed_forward(w1, w2, w3, b1, b2, b3, a0)
    error = cost(y_hat, y)
    {dC_dW3, dC_db3, dC_dA2} = backprop_layer_3(y_hat, y, m, a2, w3)
    {dC_dW2, dC_db2, dC_dA1} = backprop_layer_2(dC_dA2, a1, a2, w2, m)
    {dC_dW1, dC_db1} = backprop_layer_1(dC_dA1, a1, a0, w1, m)

    # Update weights and biases
    w3 = Nx.subtract(w3, Nx.multiply(alpha, dC_dW3))
    w2 = Nx.subtract(w2, Nx.multiply(alpha, dC_dW2))
    w1 = Nx.subtract(w1, Nx.multiply(alpha, dC_dW1))
    b3 = Nx.subtract(b3, Nx.multiply(alpha, dC_db3))  # Fix: Use b3
    b2 = Nx.subtract(b2, Nx.multiply(alpha, dC_db2))  # Fix: Use b2
    b1 = Nx.subtract(b1, Nx.multiply(alpha, dC_db1))  # Fix: Use b1
    cond do
      rem(i, 20) == 0 -> IO.puts("Error: #{Nx.to_number(error)}")
      IO.inspect(w3, label: "Updated w3")
IO.inspect(b3, label: "Updated b3")
      true -> :ok
    end
    {w3, w2, w1, b3, b2, b1}
  end

  def sigmoid(arr) do
    Nx.divide(1, Nx.add(1, Nx.exp(Nx.multiply(arr, -1))))
  end
  def prepare_data(n) do
    x = Nx.tensor([
      [150, 70],
      [254, 73],
      [312, 68],
      [120, 60],
      [154, 61],
      [212, 65],
      [216, 67],
      [145, 67],
      [184, 64],
      [130, 69]
  ])
  m = 10
  y = Nx.tensor([0,1,1,0,0,1,1,0,1,0])
  y = Nx.reshape(y, {1, m})

  a0 = Nx.transpose(x)
  {a0, y, m}
  end

  def feed_forward(w1, w2, w3, b1, b2, b3, a0) do
    # Use Nx.dot for matrix multiplication
    z1 = Nx.dot(w1, a0) |> Nx.add(b1)
    a1 = sigmoid(z1)
    z2 = Nx.dot(w2, a1) |> Nx.add(b2)
    a2 = sigmoid(z2)
    z3 = Nx.dot(w3, a2) |> Nx.add(b3)  # Fixed b2 to b3
    y_hat = sigmoid(z3)
    {y_hat, a1, a2}


end
def cost(y_hat, y) do
  y_hat = Nx.clip(y_hat, 1.0e-7, 1 - 1.0e-7)  # Prevent log(0) error
  losses = Nx.multiply(-1, Nx.add(
    Nx.multiply(y, Nx.log(y_hat)),
    Nx.multiply(Nx.subtract(1, y), Nx.log(Nx.subtract(1, y_hat)))
  ))
  m = elem(Nx.shape(Nx.flatten(y_hat)), 0)
  summed_losses = Nx.divide(Nx.sum(losses, axes: [1]), m)
  Nx.sum(summed_losses)
end


def backprop_layer_3(y_hat, y, m, a2, w3) do
  a3 = y_hat
  dC_dZ3 = Nx.subtract(a3, y)  # Fix: Correct derivative
  dZ3_dW3 = a2
  dC_dW3 = Nx.dot(dC_dZ3, Nx.transpose(dZ3_dW3)) |> Nx.divide(m)
  dC_db3 = Nx.sum(dC_dZ3, axes: [1]) |> Nx.divide(m)
  dZ3_dA2 = w3
  dC_dA2 = Nx.dot(Nx.transpose(w3), dC_dZ3)
  {dC_dW3, dC_db3, dC_dA2}
end
def backprop_layer_2(propogator_dC_dA2, a1, a2, w2, m) do
  dA2_dZ2 = Nx.subtract(1, a2) |> Nx.multiply(a2)  # Derivative of sigmoid
  dC_dZ2 = Nx.multiply(propogator_dC_dA2, dA2_dZ2)  # Fix: Correct derivative
  dZ2_dW2 = a1
  dC_dW2 = Nx.dot(dC_dZ2, Nx.transpose(dZ2_dW2)) |> Nx.divide(m)
  dC_db2 = Nx.sum(dC_dZ2, axes: [1]) |> Nx.divide(m)
  dZ2_dA1 = w2
  dC_dA1 = Nx.dot(Nx.transpose(w2), dC_dZ2)
  {dC_dW2, dC_db2, dC_dA1}
end
def backprop_layer_1(propogator_dC_dA1, a1, a0, w1, m) do
  dA1_dZ1 = Nx.subtract(1, a1) |> Nx.multiply(a1)  # Derivative of sigmoid
  dC_dZ1 = Nx.multiply(propogator_dC_dA1, dA1_dZ1)  # Fix: Correct derivative
  dZ1_dW1 = a0
  dC_dW1 = Nx.dot(dC_dZ1, Nx.transpose(dZ1_dW1)) |> Nx.divide(m)
  dC_db1 = Nx.sum(dC_dZ1, axes: [1]) |> Nx.divide(m)
  {dC_dW1, dC_db1}
end
end
NeuralNetwork.trying()
