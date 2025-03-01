import torch

def Guassian(A):
    assert A.shape[0]+1==A.shape[1],"Matrix rank mismatch"
    n=A.shape[0]

    for i in range(n):
        max_row_rel=torch.argmax(torch.abs(A[i:,i]))
        max_row=max_row_rel.item()+i
        A[[i,max_row]]=A[[max_row,i]]
        if A[i,i]==0:
            raise ValueError("Singular")

        for row in range(i+1,n):
            m=A[row,i]/A[i,i]
            A[row]=A[row]-m*A[i]

    x=torch.zeros(n)
    for i in range(n-1,-1,-1):
        sum_val=torch.dot(A[i,i+1:n],x[i+1:n])
        x[i]=(A[i,-1]-sum_val)/A[i,i]

    return x

if __name__=="__main__":
    device=torch.device("cuda")
    A=torch.tensor([
    [4,1,1,1],
    [1,5,2,8],
    [1,1,-1,1]
    ],dtype=torch.float32)
    A.to(device)

    solution=Guassian(A)

    print(solution)
