def gen_chained_data1(seq_len, size, save_root, threshold=0.9, mode="xor"):
    import os
    import torch
    
    # generate X
    X = torch.rand((size, seq_len))
    X[X>=threshold] = 1
    X[X<threshold] = 0
    X = X.int()
    
    # generate Y
    Y = torch.sum(X, dim=1)
    if mode == "xor":
        Y = Y % 2 != 0
    elif mode == "disj":
        Y = Y != 0
    elif mode == "conj":
        Y = torch.ones_like(Y) * seq_len == Y
    Y = Y.int().unsqueeze(dim=1)
    
    # save data
    save_dir = os.path.join(save_root, str(seq_len))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    torch.save(X, os.path.join(save_dir, "features.pt"))
    torch.save(Y, os.path.join(save_dir, "labels.pt"))
    
    return X, Y


def gen_chained_data2(seq_len, size, save_root, test_ratio=0.1, ext_ratio=0.02, mode="xor"):
    import os
    import torch
    
    def get_extreme_data(seq_len, ext_size, ext_len=10):
        import numpy as np
        import torch
        X = []
        num = int(ext_size / (2*ext_len) )

        for i in range(ext_len):
            # generate almost 1 case where 1:0 = (seq_len-i, i)
            x10 = []
            for _ in range(num):
                a = [1 for _ in range(seq_len-i)]+[0 for _ in range(i)] 
                np.random.shuffle(a)
                x10.append(a)
            x10 = torch.Tensor(x10)
            X.append(x10)
            # generate almost 0 case where 0:1 = (seq_len-i, i)
            x01 = []
            for _ in range(num):
                a = [1 for _ in range(i)]+[0 for _ in range(seq_len-i)] 
                np.random.shuffle(a)
                x01.append(a)
            X.append(torch.Tensor(x01))

        X = torch.concatenate(X, dim=0)
        X = X[torch.randperm(ext_size)]

        return X
    
    # generate X
    test_size = int(size * test_ratio)
    train_size = size - test_size
    ext_test_size = int(test_size * ext_ratio)
    ext_train_size = int(train_size * ext_ratio)
    
    X1r = torch.rand((train_size-ext_train_size, seq_len))
    X1e = get_extreme_data(seq_len, ext_train_size)
    X1 = torch.concatenate([X1r,X1e], dim=0)
    X1 = X1[torch.randperm(train_size)]
    
    X2r = torch.rand((test_size-ext_test_size, seq_len))
    X2e = get_extreme_data(seq_len, ext_test_size)
    X2 = torch.concatenate([X2r,X2e], dim=0)
    X2 = X2[torch.randperm(test_size)]
    
    X = torch.concatenate([X1, X2], dim=0)
    X[X>=0.5] = 1
    X[X<0.5] = 0
    X = X.int()
    
    # generate Y
    Y = torch.sum(X, dim=1)
    if mode == "xor":
        Y = Y % 2 != 0
    elif mode == "disj":
        Y = Y != 0
    elif mode == "conj":
        Y = torch.ones_like(Y) * seq_len == Y
    Y = Y.int().unsqueeze(dim=1)
    
    # save data
    save_dir = os.path.join(save_root, str(seq_len))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    torch.save(X, os.path.join(save_dir, "features.pt"))
    torch.save(Y, os.path.join(save_dir, "labels.pt"))
    
    return X, Y