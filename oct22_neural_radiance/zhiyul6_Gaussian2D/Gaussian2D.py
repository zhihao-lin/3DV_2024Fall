import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt


class Gaussian2DModel:
    def __init__(self, numGaussians, targetImgNormalized):
        """
        numGaussians 初始 2D Gaussian 数量
        targetImg 经过归一化的
        """
        self.device = self.chooseDevice()
        
        # 初始化高斯参数
        imgHeight, imgWidth, imgChannel = targetImgNormalized.shape
        print("targetImgNormalized.shape : ", targetImgNormalized.shape)
        
        self.targetImgNormalized = torch.tensor(targetImgNormalized, dtype=torch.float32).to(self.device)
        
        self.numGaussians = numGaussians
        self.imgHeight = imgHeight
        self.imgWidth = imgWidth
        
        self.sigmaRange = 10

        # np.random.rand 生成的随机数在 [0, 1) 之间
        self.x0 = torch.tensor(np.random.rand(numGaussians) * self.imgHeight, requires_grad=True, device=self.device)
        self.y0 = torch.tensor(np.random.rand(numGaussians) * self.imgWidth, requires_grad=True, device=self.device)
        
        self.sigmaX = torch.tensor(np.random.rand(numGaussians) * self.sigmaRange, requires_grad=True, device=self.device)
        self.sigmaY = torch.tensor(np.random.rand(numGaussians) * self.sigmaRange, requires_grad=True, device=self.device)
        
        self.theta = torch.tensor(np.random.rand(numGaussians) * 2* np.pi - np.pi, requires_grad=True, device=self.device)
        
        self.cR = torch.tensor(np.random.rand(numGaussians), requires_grad=True, device=self.device)
        self.cG = torch.tensor(np.random.rand(numGaussians), requires_grad=True, device=self.device)
        self.cB = torch.tensor(np.random.rand(numGaussians), requires_grad=True, device=self.device)
        
        self.alpha = torch.tensor(np.random.rand(numGaussians) * 0.25, requires_grad=True, device=self.device)  # 透明度
        
        self.rowGrid = torch.arange(0, imgHeight).unsqueeze(1).repeat(1, imgWidth).to(self.device)  # 创建一个 2D 网格
        # print(x.shape)
        self.colGrid = torch.arange(0, imgWidth).unsqueeze(0).repeat(imgHeight, 1).to(self.device)
        # print(y.shape)
        self.renderImgTorch = None

        self.lrCoeff = 5
        # 定义优化器
        self.optimizer = torch.optim.Adam([
            {'params': self.x0, 'lr': 0.01 * self.lrCoeff},
            {'params': self.y0, 'lr': 0.01 * self.lrCoeff},
            {'params': self.sigmaX, 'lr': 0.02 * self.lrCoeff},
            {'params': self.sigmaY, 'lr': 0.02 * self.lrCoeff},
            {'params': self.theta, 'lr': 0.005 * self.lrCoeff},
            {'params': self.cR, 'lr': 0.02 * self.lrCoeff},
            {'params': self.cG, 'lr': 0.02 * self.lrCoeff},
            {'params': self.cB, 'lr': 0.02 * self.lrCoeff},
            {'params': self.alpha, 'lr': 0.01 * self.lrCoeff},
        ])

    # def matplotlibShowRender(self):
    #     imgBGR = self.renderImgTorch.detach().cpu().numpy()
    #     imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
    #     plt.imshow(imgRGB)
    #     plt.axis('off')  # 关闭坐标轴
    #     plt.show()

    def chooseDevice(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using Device : ", device)
        return device
        

    def lossAndBackwardAndStep(self):
        # Loss
        self.render()
        loss = torch.nn.functional.mse_loss(self.renderImgTorch, self.targetImgNormalized)
        
        self.optimizer.zero_grad()  # 在梯度下降过程中，每次更新参数后都需要将梯度清零，否则梯度会累加
        loss.backward()  # 反向传播，计算梯度。将损失函数对所有参数的梯度计算出来并存储在参数的.grad属性中
        self.optimizer.step()  # 更新参数，调用优化器的step方法，将参数的值更新为param - lr * grad
        
        return loss.item()  # 返回当前损失值


    def render(self):
        """
        x 和 y 是 2D 网格矩阵，用于表示图像中所有像素的位置，方便批量处理
        """
        # with torch.no_grad():
        x = self.rowGrid
        y = self.colGrid
        
        imageR = torch.zeros_like(x, dtype=torch.float32)
        imageG = torch.zeros_like(x, dtype=torch.float32)
        imageB = torch.zeros_like(x, dtype=torch.float32)

        # 计算每个高斯对图像的贡献
        for i in range(self.numGaussians):
                            
            cos_theta = torch.cos(self.theta[i])
            sin_theta = torch.sin(self.theta[i])
            
            R = torch.tensor([[cos_theta, -sin_theta], 
                                [sin_theta,  cos_theta]], dtype=torch.float32).to(self.device)
            
            

            deltaX = (x - self.x0[i]).to(torch.float32)
            deltaY = (y - self.y0[i]).to(torch.float32)
        
            
            rotatedCoords = R.T @ torch.stack([deltaX.flatten(), deltaY.flatten()]).reshape(2, -1)  #   # 高斯椭球旋转 R，相当于坐标系的点旋转 R^T
            dxRot = rotatedCoords[0].reshape(x.shape)
            dyRot = rotatedCoords[1].reshape(x.shape)
            
            # 判断是否收到椭球影响(3σ)
            distNormalizedSquared = (dxRot ** 2) / (self.sigmaX[i] ** 2) + (dyRot ** 2) / (self.sigmaY[i] ** 2)
            sigmaMultiplier = 3
            effectedMask = distNormalizedSquared <= (sigmaMultiplier ** 2)
            
            G = torch.zeros_like(x, dtype=torch.float32)

            G[effectedMask] = self.alpha[i] * torch.exp(- distNormalizedSquared[effectedMask] / 2)  # 考虑是否需要clamp

            imageR += G * self.cR[i]
            imageG += G * self.cG[i]
            imageB += G * self.cB[i]
        
        self.renderImgTorch = torch.stack([imageB, imageG, imageR], dim=2)
            
        return self.renderImgTorch

        """
        # # 向量化计算
        # cos_theta = torch.cos(self.theta)
        # sin_theta = torch.sin(self.theta)

        # R = torch.stack([
        #     torch.stack([cos_theta, -sin_theta], dim=1),
        #     torch.stack([sin_theta, cos_theta], dim=1)
        # ])  # (2, 2, numGaussians)

        # deltaX = x.unsqueeze(0) - self.x0.unsqueeze(1).unsqueeze(2)
        # deltaY = y.unsqueeze(0) - self.y0.unsqueeze(1).unsqueeze(2)

        # rotatedCoords = torch.einsum('ijk,jkl->ikl', R, torch.stack([deltaX, deltaY], dim=0))
        # dxRot = rotatedCoords[0]
        # dyRot = rotatedCoords[1]

        # G = self.alpha.unsqueeze(1).unsqueeze(2) * torch.exp(
        #     -((dxRot ** 2) / (2 * self.sigmaX.unsqueeze(1).unsqueeze(2) ** 2) + 
        #     (dyRot ** 2) / (2 * self.sigmaY.unsqueeze(1).unsqueeze(2) ** 2))
        # )

        # image_r = (G * self.cR.unsqueeze(1).unsqueeze(2)).sum(dim=0)
        # image_g = (G * self.cG.unsqueeze(1).unsqueeze(2)).sum(dim=0)
        # image_b = (G * self.cB.unsqueeze(1).unsqueeze(2)).sum(dim=0)

        # return torch.stack([image_r, image_g, image_b], dim=2)
        """

    def renderWithEllipse(self, ellipse_params):
        """
        使用给定的椭圆参数进行渲染。
        ellipse_params 应该是一个包含 (x0, y0, sigmaX, sigmaY, theta, alpha, cR, cG, cB) 的列表或元组。
        """
        
        with torch.no_grad():
            x = self.rowGrid
            y = self.colGrid
            
            imageR = torch.zeros_like(x, dtype=torch.float32)
            imageG = torch.zeros_like(x, dtype=torch.float32)
            imageB = torch.zeros_like(x, dtype=torch.float32)

            for params in ellipse_params:
                x0, y0, sigmaX, sigmaY, theta, cR, cG, cB, alpha = params
                
                theta = torch.tensor(theta, dtype=torch.float32)

                cos_theta = torch.cos(theta)
                sin_theta = torch.sin(theta)
                R = torch.tensor([[cos_theta, -sin_theta], 
                                  [sin_theta,  cos_theta]]).to(self.device)

                deltaX = (x - x0).to(torch.float32)
                deltaY = (y - y0).to(torch.float32)
                
                print(deltaX.shape, deltaY.shape)
                print(torch.stack([deltaX.flatten(), deltaY.flatten()]).shape)
                a = torch.stack([deltaX, deltaY], dim=0)
                print(a.shape)
                print(R.shape)
                

                rotatedCoords = R.T @ torch.stack([deltaX.flatten(), deltaY.flatten()]).reshape(2, -1)  #   # 高斯椭球旋转 R，相当于坐标系的点旋转 R^T
                dxRot = rotatedCoords[0].reshape(x.shape)
                dyRot = rotatedCoords[1].reshape(x.shape)

                distNormalizedSquared = (dxRot ** 2) / (sigmaX ** 2) + (dyRot ** 2) / (sigmaY ** 2)
                sigmaMultiplier = 3
                effectedMask = distNormalizedSquared <= (sigmaMultiplier ** 2)

                # alpha = torch.tensor(alpha, dtype=torch.float32)
                
                G = torch.zeros_like(x, dtype=torch.float32)
                if effectedMask.any():
                    G[effectedMask] = alpha * torch.exp(-torch.clamp(distNormalizedSquared[effectedMask] / 2, max=50))

                imageR += G * cR
                imageG += G * cG
                imageB += G * cB

            return torch.stack([imageB, imageG, imageR], dim=2)
    
    def densifyGaussianAtBigDiffList(self, pickTopDiffNums):
        print("densifyGaussianAtBigDiffList add 2D Gaussian Ellipse : ", pickTopDiffNums)
        diffTorch = (self.targetImgNormalized - self.renderImgTorch).detach()
        diffAbsSumTorch = torch.sum(torch.abs(diffTorch), dim=-1)
        
        topDiffAbsVals, topDiffAbsIndex = torch.topk(diffAbsSumTorch.view(-1), pickTopDiffNums)  # view 扁平化成一维数组，以便进行排序
        
        new_x0 = (topDiffAbsIndex // self.imgWidth).float().requires_grad_().to(self.device)
        new_y0 = (topDiffAbsIndex %  self.imgWidth).float().requires_grad_().to(self.device)
        
        # print(new_x0)
        # print(new_y0)
        
        # TODO 不随机初始化方向 & 方差
        new_theta = torch.tensor(np.random.rand(pickTopDiffNums) * 2 * np.pi - np.pi, requires_grad=True, device=self.device)
        new_alpha = torch.tensor(np.random.rand(pickTopDiffNums) * 0.5, requires_grad=True, device=self.device)
        
        new_sigmaX = torch.tensor(np.random.rand(pickTopDiffNums) * self.sigmaRange, requires_grad=True, device=self.device)
        new_sigmaY = torch.tensor(np.random.rand(pickTopDiffNums) * self.sigmaRange, requires_grad=True, device=self.device)
        
        # 内部存储都是 BGR
        new_cB = diffTorch[new_x0.int(), new_y0.int(), 0].requires_grad_().to(self.device)
        new_cG = diffTorch[new_x0.int(), new_y0.int(), 1].requires_grad_().to(self.device)
        new_cR = diffTorch[new_x0.int(), new_y0.int(), 2].requires_grad_().to(self.device)
        
        self.x0 = torch.cat((self.x0.detach(), new_x0.detach())).requires_grad_()
        self.y0 = torch.cat((self.y0.detach(), new_y0.detach())).requires_grad_()
        self.sigmaX = torch.cat((self.sigmaX.detach(), new_sigmaX.detach())).requires_grad_()
        self.sigmaY = torch.cat((self.sigmaY.detach(), new_sigmaY.detach())).requires_grad_()
        self.theta = torch.cat((self.theta.detach(), new_theta.detach())).requires_grad_()
        self.cR = torch.cat((self.cR.detach(), new_cR.detach())).requires_grad_()
        self.cG = torch.cat((self.cG.detach(), new_cG.detach())).requires_grad_()
        self.cB = torch.cat((self.cB.detach(), new_cB.detach())).requires_grad_()
        self.alpha = torch.cat((self.alpha.detach(), new_alpha.detach())).requires_grad_()
        
        self.numGaussians += pickTopDiffNums
       
        # torch.cuda.empty_cache()  # 清理 GPU 的显存缓存
        
        self.optimizer = torch.optim.Adam([
            {'params': self.x0, 'lr': 0.01 * self.lrCoeff},
            {'params': self.y0, 'lr': 0.01 * self.lrCoeff},
            {'params': self.sigmaX, 'lr': 0.001 * self.lrCoeff},
            {'params': self.sigmaY, 'lr': 0.001 * self.lrCoeff},
            {'params': self.theta, 'lr': 0.005 * self.lrCoeff},
            {'params': self.cR, 'lr': 0.02 * self.lrCoeff},
            {'params': self.cG, 'lr': 0.02 * self.lrCoeff},
            {'params': self.cB, 'lr': 0.02 * self.lrCoeff},
            {'params': self.alpha, 'lr': 0.01 * self.lrCoeff},
        ])




    # def init_grad_accumulation(self):
    #     """
    #     初始化梯度累积和归一化变量
    #     """
    #     self.xyz_gradient_accum = torch.zeros_like(self.x0, device=self.x0.device)
    #     self.denom = torch.zeros_like(self.x0, device=self.x0.device)

    # def accumulate_gradients(self, gradients, mask):
    #     """
    #     累积梯度
    #     gradients: 当前计算的梯度
    #     mask: 用于标记哪些高斯点需要累积梯度
    #     """
    #     self.xyz_gradient_accum[mask] += gradients[mask]
    #     self.denom[mask] += 1


    # def densifySplit(self, grads, grad_threshold, scene_extent, N=2):
    #     """
    #     基于梯度和尺度条件进行分裂操作
    #     grad_threshold: 分裂的梯度阈值
    #     scene_extent: 场景范围，用于尺度判断
    #     N: 每个高斯点分裂成的数量
    #     """
        
    #     # 3d gaussian 原文 使用了 padded_grad 暂时不知道为什么 - TODO
        
    #     # 筛选出满足分裂条件的高斯点
    #     max_scaling = torch.max(torch.stack([self.sigmaX, self.sigmaY], dim=1), dim=1).values
    #     scaling_threshold = self.percent_dense * scene_extent

    #     # 分裂条件：梯度超过阈值且尺度小于阈值
    #     split_mask = (grads >= grad_threshold) & (max_scaling > scaling_threshold)

    #     if split_mask.any():
    #         stds_x = self.sigmaX[split_mask].repeat(N)
    #         stds_y = self.sigmaY[split_mask].repeat(N)

    #         # 在局部范围内生成新点
    #         new_x0 = self.x0[split_mask].repeat(N) + torch.randn_like(stds_x) * stds_x
    #         new_y0 = self.y0[split_mask].repeat(N) + torch.randn_like(stds_y) * stds_y

    #         # 缩小标准差以提高拟合精度
    #         new_sigmaX = stds_x * 0.8
    #         new_sigmaY = stds_y * 0.8
    #         new_alpha = self.alpha[split_mask].repeat(N) * 0.5

    #         # 合并新点和旧点
    #         self.x0 = torch.cat([self.x0, new_x0])
    #         self.y0 = torch.cat([self.y0, new_y0])
    #         self.sigmaX = torch.cat([self.sigmaX, new_sigmaX])
    #         self.sigmaY = torch.cat([self.sigmaY, new_sigmaY])
    #         self.alpha = torch.cat([self.alpha, new_alpha])

    #         # 重置累积梯度和归一化变量
    #         self.init_grad_accumulation()
            
    # def densifyClone(self, grads, grad_threshold, scene_extent):
    #     """
    #     基于梯度和尺度条件进行克隆操作
    #     grad_threshold: 克隆的梯度阈值
    #     scene_extent: 场景范围，用于尺度判断
    #     """

    #     # 筛选出满足克隆条件的高斯点
    #     max_scaling = torch.max(torch.stack([self.sigmaX, self.sigmaY], dim=1), dim=1).values
    #     scaling_threshold = self.percent_dense * scene_extent

    #     # 克隆条件：梯度超过阈值且尺度小于阈值
    #     clone_mask = (grads >= grad_threshold) & (max_scaling <= scaling_threshold)

    #     if clone_mask.any():
    #         new_x0 = self.x0[clone_mask]
    #         new_y0 = self.y0[clone_mask]
    #         new_sigmaX = self.sigmaX[clone_mask]
    #         new_sigmaY = self.sigmaY[clone_mask]
    #         new_alpha = self.alpha[clone_mask]

    #         # 合并新点和旧点
    #         self.x0 = torch.cat([self.x0, new_x0])
    #         self.y0 = torch.cat([self.y0, new_y0])
    #         self.sigmaX = torch.cat([self.sigmaX, new_sigmaX])
    #         self.sigmaY = torch.cat([self.sigmaY, new_sigmaY])
    #         self.alpha = torch.cat([self.alpha, new_alpha])

    #         # 重置累积梯度和归一化变量
    #         self.init_grad_accumulation()

    # def densifyAndPrune(self, num_new_gaussians):
    #     grads = self.xyz_gradient_accum / self.denom  # 计算平均梯度，避免极端情况
    #     grads[grads.isnan()] = 0.0
        
    #     self.densifySplit(grads, max_grad, extent)
    #     self.densifyClone(grads, max_grad, extent)
        
    #     torch.cuda.empty_cache()  # 清理 GPU 的显存缓存
        
        

if __name__ == '__main__':    
    ellipse_params = [
        (100, 100, 10, 20, torch.pi / 6, 1.0, 0.0, 0.0, 1.0),  # 一个红色椭圆
        (100, 120, 20, 10, 0, 0.0, 1.0, 0.0, 0.5)             # 一个绿色椭圆
    ]

    imgHeight=128
    imgWidth=256
    originImg = np.zeros((imgHeight, imgWidth, 3), dtype=np.float32)
    
    
    
    model = Gaussian2DModel(numGaussians=len(ellipse_params), targetImgNormalized=originImg)


    
    x = torch.arange(0, imgHeight).unsqueeze(1).repeat(1, imgWidth)  # 创建一个 2D 网格
    print(x.shape)
    y = torch.arange(0, imgWidth).unsqueeze(0).repeat(imgHeight, 1)
    print(y.shape)
    
    image = model.renderWithEllipse(ellipse_params)
    image_np = image.detach().cpu().numpy()
    
    # image_np = image.detach().cpu().numpy() * 255
    # image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    
    # image_bgr = cv2.merge((image_np[:, :, 2], image_np[:, :, 1], image_np[:, :, 0]))  # OpenCV 使用 BGR 格式
    
    # 显示图像
    cv2.imshow('Gaussian Render', image_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

