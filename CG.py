import torch 
def generate_spd_ortho(n, device='cuda'):
    # 生成随机正交矩阵（QR分解实现）
    A = torch.randn(n, n, device=device)
    Q, _ = torch.linalg.qr(A)  
    
    # 生成正对角阵（指数分布避免过大特征值）
    D = torch.diag(torch.rand(n, device=device) * 10 + 1e-3)
    
    # 构造正定矩阵
    return Q @ D @ Q.T


def CG(A,B,max_iteration,precise,device='cuda'):
    assert A.shape[0]==A.shape[1],"Dimension Mismatch"
    n=A.shape[0]
    x=torch.randn(n,device=device)
    grad=B-torch.matmul(A,x)
    i=0
    while i <=max_iteration:
        alpha=(torch.dot(grad,grad))/(torch.dot(torch.matmul(A,grad),grad))
        x=x+torch.mul(alpha,grad)
        grad=grad-torch.mul(alpha,torch.matmul(A,grad))
        if torch.max(torch.abs(grad))<=precise:
            break
        i+=1
    return x,i


if __name__ == "__main__":
    device=torch.device('cuda')
    n=1000
    A=generate_spd_ortho(n,device=device)
    B=torch.randn(n,device=device)
    print(CG(A,B,100000,1e-5,device))


    
