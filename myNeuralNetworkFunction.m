function [y1] = myNeuralNetworkFunction(x1)
%MYNEURALNETWORKFUNCTION neural network simulation function.
%
%
% [y1] = myNeuralNetworkFunction(x1) takes these arguments:
%   x = Qx2 matrix, input #1
% and returns:
%   y = Qx1 matrix, output #1
% where Q is the number of samples.

% ===== NEURAL NETWORK CONSTANTS =====

% Input 1
x1_step1.xoffset = [278.15;0];
x1_step1.gain = [0.0176803394625177;0.746268656716418];
x1_step1.ymin = -1;

% Layer 1
b1 = [-0.34627348777537741986;0.19268932852845949144;-4.4127869652329323458];
IW1_1 = [1.565646046841941974 -1.2367576306241330197;-0.90218835576475486793 -0.56493237450086086771;-4.5235897440481194209 0.085272171971478530339];

% Layer 2
b2 = -0.27223917946288078706;
LW2_1 = [-0.41259691298857592567 1.0945958755436517862 0.14745301207268884935];

% Output 1
y1_step1.ymin = -1;
y1_step1.gain = 21.0084033613445;
y1_step1.xoffset = 1.2902;

% ===== SIMULATION ========

% Dimensions
Q = size(x1,1); % samples

% Input 1
x1 = x1';
xp1 = mapminmax_apply(x1,x1_step1);

% Layer 1
a1 = tansig_apply(repmat(b1,1,Q) + IW1_1*xp1);

% Layer 2
a2 = repmat(b2,1,Q) + LW2_1*a1;

% Output 1
y1 = mapminmax_reverse(a2,y1_step1);
y1 = y1';
end

% ===== MODULE FUNCTIONS ========

% Map Minimum and Maximum Input Processing Function
function y = mapminmax_apply(x,settings)
y = bsxfun(@minus,x,settings.xoffset);
y = bsxfun(@times,y,settings.gain);
y = bsxfun(@plus,y,settings.ymin);
end

% Sigmoid Symmetric Transfer Function
function a = tansig_apply(n,~)
a = 2 ./ (1 + exp(-2*n)) - 1;
end

% Map Minimum and Maximum Output Reverse-Processing Function
function x = mapminmax_reverse(y,settings)
x = bsxfun(@minus,y,settings.ymin);
x = bsxfun(@rdivide,x,settings.gain);
x = bsxfun(@plus,x,settings.xoffset);
end
