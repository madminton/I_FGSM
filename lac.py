import torch
import torchvision.transforms as transforms
import cv2  
import legacy  
import dnnlib
import matplotlib.pyplot as plt
import numpy as np

# 가중치 불러오기 (StyleGAN 가중치)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with dnnlib.util.open_url('stylegan2-ada-pytorch/ffhq.pkl') as f:
    network_data = legacy.load_network_pkl(f)  
    G = network_data['G_ema'].to(device)  
    D = network_data['D'].to(device)  
# I-FGSM
def apply_ifgsm(model, x, y, epsilon=0.01, alpha=0.001, iterations=10):
    x_adv = x.clone().detach().requires_grad_(True).to(device)
    
    for i in range(iterations):
        c = None  
        outputs = model(x_adv, c)  
        print(f"Outputs shape: {outputs.shape}")         
        loss = torch.nn.functional.cross_entropy(outputs, y)  
        model.zero_grad()
        loss.backward() 
        grad = x_adv.grad.sign()  
        x_adv = x_adv + alpha * grad 
        eta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon) 
        x_adv = torch.clamp(x + eta, min=0, max=1).detach()  
        x_adv.requires_grad_()  
        
    return x_adv
# 이미지 로드 및 전처리 
def load_image(image_filename):
    img = cv2.imread(image_filename)  
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    img = cv2.resize(img, (1024, 1024))  
    img = img / 255.0  
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)  
    return img.float()

image_filename = 'C:/Users/samsung/OneDrive/Desktop/taco ex/stylegan2-ada-pytorch/Vangogh.png'  # 같은 디렉토리의 이미지 파일

# 이미지 로드
img = load_image(image_filename)

# I-FGSM 적용
epsilon = 1  # 노이즈 강도，원래 0.01
alpha = 1  # 학습 속도,원래 0.001
iterations = 1  # 반복 횟수,원래10

img = img.to(device)  # 이미지를 장치로 이동
y = torch.tensor([0]).to(device)  # 0 또는 1로 타겟 클래스 설정

# StyleGAN의 판별기(D)를 사용해 I-FGSM 적용
adv_img = apply_ifgsm(D, img, y, epsilon, alpha, iterations)

# 이미지 시각화
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img.squeeze().permute(1, 2, 0).cpu().numpy())

plt.subplot(1, 2, 2)
plt.title("Adversarial Image")
plt.imshow(adv_img.squeeze().permute(1, 2, 0).cpu().detach().numpy())

plt.show()


