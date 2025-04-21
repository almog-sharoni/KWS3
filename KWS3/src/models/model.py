class KeywordSpottingModel_with_cls(nn.Module):
    def __init__(self, num_classes, num_mamba_layers=2, dropout_rate=0.5, use_cls_token=False):
        super(KeywordSpottingModel_with_cls, self).__init__()
        self.num_classes = num_classes
        self.num_mamba_layers = num_mamba_layers
        self.dropout_rate = dropout_rate
        self.use_cls_token = use_cls_token
        
        # Define model layers here
        self.layers = nn.ModuleList()
        for _ in range(num_mamba_layers):
            self.layers.append(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.fc = nn.Linear(32 * (input_size // 2**num_mamba_layers), num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(x)
        x = self.fc(x)
        
        return x