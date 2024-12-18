# Binary-Classification-in-PyTorch
 It predicts whether a data point (from a synthetic dataset) belongs to one of two classes, represented by "red" and "blue" dots.

*Generate data*
X, y = make_circles(n_samples, noise=0.03, random_state=42)

- make_circles creates a dataset of concentric circles for classification.
  X: 2D features (coordinates of the points).
  y: Labels (0 or 1 for the two classes)

*Prepare data*
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

- The data is converted to PyTorch tensors.
- The dataset is split into training and testing subsets (80% training, 20% testing).

*Define neural network*

```
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)
    def forward(self, x):
        return self.layer_2(self.layer_1(x))'
```
- The model uses two linear layers:
    layer_1: Upscales 2 input features to 5 features.
    layer_2: Reduces 5 features to 1 output.
    The forward method defines how data passes through the network.

*Loss function & Optimizer*
loss_fn = nn.BCEWithLogitsLoss()  # Combines sigmoid activation and binary cross-entropy loss
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

- Loss Function: BCEWithLogitsLoss is suitable for binary classification and applies sigmoid activation internally.
- Optimizer: Stochastic Gradient Descent (SGD) updates model weights.

*Training Loop*
for epoch in range(epochs):
    model_0.train()
    y_logits = model_0(X_train).squeeze()
    y_preds = torch.round(torch.sigmoid(y_logits))
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_train, y_preds)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_0.eval()
    with torch.inference_mode():
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_test, test_pred)
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")

- The training loop runs for 100 epochs:

   Training Phase:
   
   Forward pass: Predict logits using the model.
   Loss calculation: Compare predictions with y_train.
   Backpropagation: Compute gradients.
   Optimizer step: Update weights.
   Evaluation Phase:
   
   Evaluate on the test set without gradient tracking.
   Compute test loss and accuracy.

*Evaluating Model*
y_logits = model_0(X_test)
y_pred_prob = torch.sigmoid(y_logits)
y_preds = torch.round(y_pred_prob)

- Sigmoid converts raw logits into probabilities.
- Rounding converts probabilities to class labels (0 or 1).

