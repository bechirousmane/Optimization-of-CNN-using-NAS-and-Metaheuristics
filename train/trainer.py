import torch.nn as nn
import torch.optim as optim
import torch

class ModelTrainer :
    def __init__(self, model, device, lr, epochs, train_loader, test_loader,optimizer):
        """
            Initializes the model trainer.
            Args :
                model : torch.nn.Sequential, the model to be trained
                device : str,  device name
                lr : float, the learning rate
                epochs : int, the number of epochs
                train_loader : DataLoader, data for training
                test_loader : DataLoader, data for testing
                optimizer : str, the optimizer name. It must be among Adam and AdamW 
        """
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = self._optimizer(optimizer)
        self.criterion = nn.CrossEntropyLoss()
        self.loss_history = []

    def _optimizer(self, optimName) :
        if optimName == "Adam" :
            return optim.Adam(self.model.parameters(), lr=self.lr)
        elif optimName == "AdamW" :
            return optim.AdamW(self.model.parameters(), lr=self.lr)
        else :
            raise ValueError("Unsupported optimizer")
        
    def train(self, verbose=False) :
        """
            Traines the model.
            Args : 
                verbose : bool, whether to print progress during training
            Return : list, the loss list
        """
        self.model.train()
        self.loss_history = []
        for epoch in range(self.epochs) :
            epoch_loss = 0

            for batch_idx, (data,targets) in enumerate(self.train_loader) :
                data, targets = data.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss/len(self.train_loader)
            if verbose :
                print(f"Epochs : [{epoch}/{self.epochs}], average loss : {avg_loss:.6f}")

            self.loss_history.append(avg_loss)

    def test(self):
        """
        Test the model on the test set.
        Returns:
            tuple: (accuracy, test_loss) where accuracy is the percentage of correct predictions
                  and test_loss is the average loss across all test batches
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in self.test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = 100 * correct / total
        avg_test_loss = test_loss / len(self.test_loader)
        
        return accuracy, avg_test_loss