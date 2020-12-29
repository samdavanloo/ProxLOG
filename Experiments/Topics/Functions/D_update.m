function [D,f] = D_update(D, An, Bn,X,A)

    [~, n] = size(D);
    f = zeros(5,1);
    for i = 1:5

        for j = 1:n
            u = D(:, j) + (Bn(:, j) - D * An(:, j)) / An(j, j);
            D(:, j) = ProjectOntoSimplex(u, 1);
        end
        
        temp = X-D*A;        
        f(i) = norm(temp,'fro')^2/2;

    end

end
