import torch

def Jacobi(A,B,init_x=None,max_iteration=1000,precise=0.01):
    assert A.shape[0]==A.shape[1],"Dimension Mismatch"
    n=A.shape[0]
    if init_x==None:
        x=torch.randn(n)
    else:
        x=init_x
    D=torch.diag(A)
    diag=torch.diag(D)
    S=diag-A
    for j in range(max_iteration):
        x_now=torch.div(torch.matmul(S,x)+B,D)
        if torch.abs(torch.max(x_now-x))<=precise:
            break
        else:
            x=x_now
        j+=1
    return x
if __name__ == "__main__":
    device=torch.device("cuda")
    A=torch.tensor(
        [
            [4,1,1],
            [1,5,2],
            [1,8,1]
        ],
        dtype=torch.float32
    )
    B=torch.tensor([1,2,3],dtype=torch.float32)
    A.to(device)
    B.to(device)
    print(Jacobi(A,B,max_iteration=1000,precise=1e-5))


