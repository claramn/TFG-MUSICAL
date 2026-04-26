import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
import torchaudio.transforms as T
from src.dataset_og import NSynth

# Assuming device is defined elsewhere, but for now, set it here
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Scheduler(nn.Module):
    def __init__(self, num_epochs, beta_init=1e-4, beta_finish=0.02, device='cpu'):
        super().__init__()
        self.beta = torch.linspace(beta_init, beta_finish, num_epochs, device=device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def forward(self, t):
        return self.beta[t], self.alpha[t], self.alpha_bar[t]

class Diffuser(nn.Module):
    def __init__(self, model, scheduler):
        super().__init__()
        self.model = model
        self.scheduler = scheduler

    # x: input image, t: time step
    def forward(self, x, t):
        e = torch.randn_like(x)  # ruido gaussiano mismo tamaño que x
        beta_t, alpha_t, alpha_bar_t = self.scheduler(t)
        alpha_bar_t = alpha_bar_t.view(-1, 1, 1, 1)
        z = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * e
        return z, e

class Embeder(nn.Module):
    def __init__(self, num_epochs, embed_dim, device):
        super().__init__()
        position = torch.arange(num_epochs, device=device).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2, device=device).float() * -(math.log(10000.0) / embed_dim))
        embeddings = torch.zeros(num_epochs, embed_dim, device=device)
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)
        self.embeddings = embeddings

    def forward(self, x, t):
        embeds = self.embeddings[t].to(x.device)
        return embeds[:, :, None, None]

class DummyLayer(nn.Module):
    def __init__(self, in_channels, out_channels, norm_groups, emb_dim, skip=False, stride=1):
        super().__init__()
        self.skip = skip
        self.stride = stride
        if stride == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.GroupNorm(norm_groups, out_channels),
                nn.ReLU()
            )
        elif stride == -1:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 3, padding=1, stride=2, output_padding=1),
                nn.GroupNorm(norm_groups, out_channels),
                nn.ReLU()
            )
        if self.skip:
            self.res_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x, t_emb):
        out = self.conv(x)
        if self.skip:
            res = self.res_conv(out)
            return out, res
        else:
            return out

class DiffusionModel(nn.Module):
    def __init__(self, layer_channels, norm_groups, up_layers, down_layers, bottleneck, embedder, input_channels=1, output_channels=1):
        super().__init__()
        self.relu = nn.ReLU()
        sin, sout = layer_channels
        self.conv_in = nn.Sequential(
            nn.Conv2d(input_channels, sin, kernel_size=3, padding=1),
            nn.GroupNorm(norm_groups, sin),
            nn.ReLU()
        )
        self.conv_out = nn.Sequential(
            nn.GroupNorm(norm_groups, sout),
            nn.ReLU(),
            nn.Conv2d(sout, output_channels, kernel_size=3, padding=1),
        )
        self.embeder = embedder
        self.down_layers = down_layers
        self.bottleneck = bottleneck
        self.up_layers = up_layers
        
    def forward(self, x, t):
        return self.forward_withres(x, t)
    
    def forward_withres(self, x, t):
        B, C, H, W = x.shape
        x = self.conv_in(x)
        t_emb = self.embeder(x, t)
        residuals = []
        for layer in self.down_layers:
            x, r = layer(x, t_emb)
            residuals.append(r)
        x = self.bottleneck(x, t_emb)
        for layer in self.up_layers:
            r = residuals.pop()
            x = torch.concat((x, r), dim=1)
            x = layer(x, t_emb)
        return self.conv_out(x)

def train_old(input_size, epochs, batch_size=16, lr=1e-3):
    layers = [
        DummyLayer(64, 128, 8).to(device),
        DummyLayer(128, 128, 8).to(device),
    ]
    embedder = Embeder(num_epochs=epochs, embed_dim=128)
    model = DiffusionModel(
        channels=[64, 128],
        norm_groups=8,
        layers=layers,
        embedder=embedder,
        input_channels=1,
        output_channels=1        
    ).to(device)
    
    best_loss = 1
    
    train_loader = DataLoader(NSynth('training'), batch_size=batch_size, shuffle=True,  pin_memory=True)
    scheduler = Scheduler(num_epochs=epochs).to(device)
    diffuser = Diffuser(model, scheduler).to(device)
    
    optimizer = torch.optim.Adam(diffuser.parameters(), lr=lr)
    
    mse_loss = nn.MSELoss()
        
    for epoch in range(epochs):
        for wave, _, _, _ in train_loader:
            wave = wave.to(device)
            x = stft_transform(wave)
            
            x = x.to(device)
            batch_size = x.size(0)
            t = torch.randint(0, epochs, (batch_size,), device=device)

            z, e = diffuser(x, t)
            e_pred = model(z, t)

            loss = mse_loss(e_pred, e)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
            
        if loss.item() < best_loss:
            best_loss = loss.item()
            print(f"Model saved at epoch {epoch} with loss {best_loss}")
    
    return model, scheduler

def train_minst(input_size, epochs, batch_size=16, lr=1e-3, path=r'C:\Users\Articuno\Desktop\TFG-info\data\mnist'):
    layers = [
        DummyLayer(64, 128, 8).to(device),
        DummyLayer(128, 128, 8).to(device),
    ]
    embedder = Embeder(num_epochs=epochs, embed_dim=128)
    model = DiffusionModel(
        channels=[64, 128],
        norm_groups=8,
        layers=layers,
        embedder=embedder,
        input_channels=1,
        output_channels=1        
    ).to(device)
    
    best_loss = 1
    
    train_ds = FashionMNIST(root=path, train=True,  download=True, transform=ToTensor())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  pin_memory=True)
    scheduler = Scheduler(num_epochs=epochs).to(device)
    diffuser = Diffuser(model, scheduler).to(device)
    
    optimizer = torch.optim.Adam(diffuser.parameters(), lr=lr)
    
    mse_loss = nn.MSELoss()
        
    for epoch in range(epochs):
        for x, _ in train_loader:
            x = x.to(device)
            batch_size = x.size(0)
            t = torch.randint(0, epochs, (batch_size,), device=device)

            z, e = diffuser(x, t)
            e_pred = model(z, t)

            loss = mse_loss(e_pred, e)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
            
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), r'C:\Users\Articuno\Desktop\TFG-info\data\models\diff_mnist.pth')
            print(f"Model saved at epoch {epoch} with loss {best_loss}")
    
    return model, scheduler

def train(input_size, epochs=1000, batch_size=16, lr=1e-3, path=r'C:\Users\Articuno\Desktop\TFG-info\data\models\diff.pth'):
    layers = [
        DummyLayer(64, 128, 8).to(device),
        DummyLayer(128, 128, 8).to(device),
    ]
    embedder = Embeder(num_epochs=epochs, embed_dim=128)
    model = DiffusionModel(
        channels=[64, 128],
        norm_groups=8,
        layers=layers,
        embedder=embedder,
        input_channels=1,
        output_channels=1        
    ).to(device)
    
    best_loss = 1
    
    train_loader = DataLoader(NSynth('training'), batch_size=batch_size, shuffle=True,  pin_memory=True)
    scheduler = Scheduler(num_epochs=epochs).to(device)
    diffuser = Diffuser(model, scheduler).to(device)
    
    optimizer = torch.optim.Adam(diffuser.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()
    mse_loss = nn.MSELoss()
        
    for epoch in range(epochs):
        for wave, _, _, _ in train_loader:
            wave = wave.to(device)
            with torch.cuda.amp.autocast():
                x = stft_transform(wave)
                batch_size = x.size(0)
                t = torch.randint(0, epochs, (batch_size,), device=device)
                z, e = diffuser(x, t)
                e_pred = model(z, t)
                loss = mse_loss(e_pred, e)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
            
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), path)
            print(f"Model saved at epoch {epoch} with loss {best_loss}")
    
    return model, scheduler

@torch.no_grad()
def sample_images_2(model, scheduler, num_images=4, image_size=(1, 28, 28)):
    model.eval()

    x = torch.randn(num_images, *image_size, device=device)
    T = scheduler.beta.size(0)

    betas = scheduler.beta
    alphas = scheduler.alpha
    alpha_bars = scheduler.alpha_bar

    for t in reversed(range(T)):
        t_batch = torch.full((num_images,), t, device=device, dtype=torch.long)

        e_pred = model(x, t_batch)
        
        beta_t = betas[t]
        alpha_t = alphas[t]
        alpha_bar_t = alpha_bars[t]

        coef1 = 1 / torch.sqrt(alpha_t)
        coef2 = beta_t / torch.sqrt(1 - alpha_bar_t)

        mu = coef1 * (x - coef2 * e_pred)

        if t > 0:
            noise = torch.randn_like(x)
            sigma_t = torch.sqrt(beta_t)
            x = mu + sigma_t * noise
        else:
            x = mu

    x = x.clamp(0, 1).cpu()
    return x

@torch.no_grad()
def sample_images(model, scheduler, num_images=4, image_size=(1, 28, 28)):
    model.eval()
    
    x = torch.randn(num_images, *image_size, device=device)
    T = scheduler.beta.size(0)
    
    for t in reversed(range(T)):
        t_batch = torch.tensor([t]*num_images, device=device)
        e_pred = model(x, t_batch)
        beta_t, alpha_t = scheduler(t_batch)
        beta_t = beta_t.view(-1, 1, 1, 1)
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        
        x = (x - torch.sqrt(beta_t) * e_pred) / torch.sqrt(alpha_t)
    
    x = x.clamp(0, 1).cpu()
    return x

# STFT transform for audio
sample_rate = 16000
n_fft = 1500
hop_length = 250
win_length = n_fft
stft_transform = T.Spectrogram(
    n_fft=n_fft, win_length=win_length, hop_length=hop_length,
    power=2.0, onesided=False, center=False
).to(device)