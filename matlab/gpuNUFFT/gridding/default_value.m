function [v] = default_value(A)
if (isempty(A) == 1)
   v = 0;
else
   v=A;
end