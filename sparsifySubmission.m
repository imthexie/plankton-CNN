X = csvread('submission0.csv', 1, 1);

X_prime = zeros(size(X,1), size(X, 2));
n = 121;
for k=1:size(X, 1)
    k
    x = X(k, :)';
    
    %Cost matrix
    C = zeros(n,n);
    for i = 1:n
        for j = 1:n
            if i == j
               continue;
            end
            C(i,j) = x(j) / x(i);        
        end
    end

    %gamma = std(x) / 0.00005;
    gamma = 1 / std(x);
    cvx_begin

    variable S(n,n) nonnegative

    minimize(ones(1, n) * C' * S * ones(n,1) + gamma * norm(S* ones(n, 1) - x))

    subject to
        S' * ones(n, 1) == x;

    cvx_end
    
    X_prime(k, :) = S* ones(n, 1);
    
end

csvwrite('submission0_sparse.csv', 1, 1);