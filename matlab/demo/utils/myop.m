function [KHK] = myop(F,FH,u_k,smaps)
[width,height,chn] = size(smaps);
u_k = reshape(u_k,width,height);

for chn_cnt = 1:chn
    s = smaps(:,:,chn_cnt);
    y(:,:,chn_cnt) = FH(F(u_k,s),s);
end
KHK = sum(y,3);
KHK = KHK(:);