function run_experiment(tensor_type, tensor_size)
    % run_tensor_experiment - Runs a CUR decomposition experiment for a specified
    % tensor type and size.
    %
    % Syntax: run_tensor_experiment('Hilbert', 500)
    %
    % Inputs:
    %   tensor_type - String specifying the type of tensor to generate
    %                 ('Toeplitz', 'Cauchy', 'Vandermonde', 'Maxwellian', 'Hilbert')
    %   tensor_size - Integer specifying the size of the tensor for the experiment

    ranks = 2:10;  % Ranks to test for decomposition
    
    % Create a log file to save results
    log_filename = sprintf('%s_results_size_%d.txt', tensor_type, tensor_size);
    log_file = fopen(log_filename, 'w');
    fprintf(log_file, 'Tensor Type: %s\n', tensor_type);
    fprintf(log_file, 'Tensor Size: %d\n\n', tensor_size);
    fprintf(log_file, 'Rank\tApproximation Error\n');
    fprintf(log_file, '-----------------------------\n');

    % Generate the tensor and reshape it into a matrix
    T = generate_4d_tensor(tensor_type, tensor_size, tensor_size, tensor_size, tensor_size);
    A = reshape(T, tensor_size * tensor_size, []);

    errors = zeros(size(ranks));  
    fprintf('starting...')
    for i = 1:length(ranks)
        r = ranks(i);
        [~, total_error] = recursive_cur_decomposition(A, tensor_size, r, r, r);
        errors(i) = total_error;

        % Log the results in the text file
        fprintf(log_file, '%d\t%.6f\n', r, total_error);
        fprintf('Completed rank %d of %d with error %.6f\n', r, ranks(end), total_error);
    end
    
    % Plot the results
    figure;
    semilogy(ranks, errors, 'o-', 'LineWidth', 2, 'MarkerSize', 8);
    xticks(ranks);  
    xlabel('Ranks', 'FontSize', 14);
    ylabel('Approximation Error (Frobenius Norm)', 'FontSize', 14);
    %title(['Error for ', tensor_type, ' Tensor, Size: ', num2str(tensor_size)], 'FontSize', 16);
    grid on;
    saveas(gcf, sprintf('%s_error_plot_size_%d.png', tensor_type, tensor_size));  % Save plot
    set(gca, 'LooseInset', get(gca, 'TightInset'));
    set(gcf, 'Color', 'w');  
    fclose(log_file);
    disp(['Experiments completed for ', tensor_type, ' tensor of size ', num2str(tensor_size), '.']);
    disp(['Results saved in ', log_filename]);
end
%% Recursive CUR Decomposition Function
function [matrices, total_error] = recursive_cur_decomposition(A, n, r1, r2, r3)
    [C, U, R] = cur_maxvol_decomposition(A, r1);
    matrices = {U};

    % Level 1: Decompose C and R
    C_reshaped = reshape(C, n, []);
    [C1, U1, R1] = cur_maxvol_decomposition(C_reshaped, r2);
    matrices{2} = C1;
    matrices{3} = U1 * R1;

    R_reshaped = reshape(R, [], n);
    [C2, U2, R2] = cur_maxvol_decomposition(R_reshaped, r3);
    matrices{4} = C2;
    matrices{5} = U2 * R2;

    % Calculate approximation and error
    Approximation = reshape(matrices{2} * matrices{3}, n * n, []) * ...
                    matrices{1} * reshape(matrices{4} * matrices{5}, [], n * n);
    total_error = norm(A - Approximation, 'fro') / norm(A, 'fro');
end

%% CUR Maxvol Decomposition Function
function [C, U, R] = cur_maxvol_decomposition(A, rank)
    [row_idx, ~] = maxvol(A, rank);
    R = A(row_idx, :);
    [col_idx, ~] = maxvol(A', rank);
    C = A(:, col_idx);
    U = pinv(A(row_idx, col_idx));
end

%% Maxvol Algorithm Function
function [row_idx, A_inv] = maxvol(A, rank)
    [n, ~] = size(A);
    row_idx = zeros(1, rank);
    rest_of_rows = 1:n;
    A_new = A;
    i = 1;

    while i <= rank && ~isempty(rest_of_rows)
        rows_norms = sum(A_new .^ 2, 2);
        [~, max_row_idx] = max(rows_norms);
        max_row = A_new(max_row_idx, :);
        row_idx(i) = rest_of_rows(max_row_idx);
        projection = (A_new * max_row') / (max_row * max_row');
        A_new = A_new - projection * max_row;

        rest_of_rows(max_row_idx) = [];
        A_new(max_row_idx, :) = [];
        i = i + 1;
    end

    row_idx = row_idx(row_idx > 0);
    A_inv = pinv(A(row_idx, :));
end

%% Tensor Generation
function T = generate_4d_tensor(type, n1, n2, n3, n4)
    switch lower(type)
        case 'toeplitz'
            T = toeplitz_tensor(n1, n2, n3, n4);
        case 'cauchy'
            T = cauchy_tensor(n1, n2, n3, n4);
        case 'vandermonde'
            T = vandermonde_tensor(n1, n2, n3, n4);
        case 'maxwellian'
            T = maxwellian_tensor(n1, n2, n3, n4);
        case 'hilbert'
            T = hilbert_tensor(n1, n2, n3, n4);
        otherwise
            error('Unknown tensor type.');
    end
end
function T = toeplitz_tensor(n1, n2, n3, n4)
    % Generate a 4D Toeplitz tensor
    c = 1:n1 * n2 * n3 * n4;
    T = toeplitz(c);
    T = reshape(T, n1, n2, n3, n4);
end

function T = cauchy_tensor(n1, n2, n3, n4)
    % 4D Cauchy tensor
    x = 1:n1 * n2;
    y = 2:n3 * n4 + 1;
    T = 1 ./ (x' + y);
    T = reshape(T, n1, n2, n3, n4);
end

function T = vandermonde_tensor(n1, n2, n3, n4)
    % 4D Vandermonde tensor
    x = linspace(0, 1, n1 * n2);
    T = vander(x);
    T = reshape(T, n1, n2, n3, n4);
end

function T = maxwellian_tensor(n1, n2, n3, n4)
    % Maxwellian Tensor Generation
    % Inputs:
    %   n1, n2, n3, n4 - dimensions of the tensor
    % Output:
    %   T - generated 4D tensor of size [n1, n2, n3, n4]
    x = linspace(-1, 1, n1);
    y = linspace(-1, 1, n2);
    vx = linspace(-1, 1, n3);
    vy = linspace(-1, 1, n4);
    T = zeros(n1, n2, n3, n4);
    for i = 1:n1
        for j = 1:n2
            for k = 1:n3
                for l = 1:n4
                    rho_x = (1 + 0.875 * sin(2 * pi * vx(k))) / (2 * sqrt(2 * pi * T_value(x(i))));
                    rho_y = (1 + 0.875 * sin(2 * pi * vy(l))) / (2 * sqrt(2 * pi * T_value(y(j))));
                    
                    T(i, j, k, l) = rho(x(i), y(j)) * (exp(-((vx(k) + 0.75)^2) / (2 * T_value(x(i)))) + exp(-((vy(l) + 0.75)^2) / (2 * T_value(y(j)))));
                end
            end
        end
    end
end

function T = T_value(w)
    T = 0.5 + 0.4 * sin(2 * pi * w);
end

function rho_val = rho(x, y)
    rho_val = (rho_component(x) / (2 * sqrt(2 * pi * T_value(x)))) + (rho_component(y) / (2 * sqrt(2 * pi * T_value(y))));
end

function rho_comp = rho_component(w)
    rho_comp = 1 + 0.875 * sin(2 * pi * w);
end

function T = hilbert_tensor(n1, n2, n3, n4)
    T = zeros(n1, n2, n3, n4);
    for i = 1:n1
        for j = 1:n2
            for k = 1:n3
                for l = 1:n4
                    T(i, j, k, l) = 1 / (i + j + k + l - 3);
                end
            end
        end
    end
end
