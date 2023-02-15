clear all;clc;close all
load HYDICE_urban
[m,n,b]=size(hsi);
XG=hsi;
%%
%% kmeans
X=reshape(XG,m*n,b);
nn=500;
cl=50;
% Perform k-Means clustering using the squared Euclidean distance metric. 
[kidx,C,~,D] = kmeans(X,cl,'MaxIter',10000); % The default distance metric is squared Euclidean distance
%Visualize the clustering.

d1=10;
d2=100;

%%
clear var_C len_C  mean_C X_temp
for i=1:cl
    i
    temp=find(kidx==i);
    len_C(i)=length(temp);
    mean_C(i,:)=mean(X(temp,:));
    for j=1:length(temp)
        X_temp=X(temp,:);
        for k=1:b
            var_C_temp(k)=var(X_temp(:,k));
        end
    end
    var_C(i)=mean(var_C_temp);
end

var_C1=[var_C;len_C];

[sx_var_C,sy_var_C]=sort(var_C,'descend');

[sx_len_C,sy_len_C]=sort(len_C);

result=[sy_var_C;sy_len_C]


%%
[rows,cols,bands]=size(hsi);
m=rows;
n=cols;
b=bands;

%% 异常点
clear new_rv
rv=sy_var_C(1:3);
rl=sy_len_C(1:3);
rvl=intersect(rv, rl)
j=1;
for i=1:1%length(rv)
    if ~isempty(intersect(rv(i), rl) )
        new_rv(j)=rv(i);
        j=j+1;
        flag=0;
    end    
end
new_rv0=new_rv;

tempA=find(kidx==new_rv(1));
anomaly=tempA;
%%
clear new_rv
new_rv=sy_var_C(end:-1:2);

%%
%%

clear chadis_tt
for i=1%1:length(tt)
    i
    %tempA=find(kidx==new_rv(i));
    tempA=find(kidx==new_rv0);
    mean_tempA=mean(X(tempA,:));
    normal_temp=[];
    Xt=X(tempA,:);
    clear dis
    for j=1:length(tempA)
        dis(j)=pdist2(mean_tempA,Xt(j,:),'euclidean');
    end
    [sx_dis,sy_dis]=sort(dis);
    chadis_tt(i)=sx_dis(end);%-sx_dis(1)
 
end
biaozhun=chadis_tt(1);

%% 异常点
biaozhun1=chadis_tt(1)*0.88;
anomaly1=[];
chadis_tt=[];
tt=new_rv;
for i=1:length(tt)
    %i
    tempA=find(kidx==tt(i));
    mean_tempA=mean(X(tempA,:));
    normal_temp=[];
    Xt=X(tempA,:);
    clear dis
    for j=1:length(tempA)
        dis(j)=pdist2(mean_tempA,Xt(j,:),'euclidean');
    end
    [sx_dis,sy_dis]=sort(dis);
        temp=find(sx_dis>=biaozhun1);
        choose=sy_dis(temp);
        anomaly1=[anomaly1;tempA(choose)]
end
anomaly=[anomaly;anomaly1];
anomaly=unique(anomaly)
%% 正常点
biaozhun2=biaozhun*0.4;
clear chadis_tt2
normal=[];

for i=1:length(new_rv)
    %i
    tempA=find(kidx==new_rv(i));
    mean_tempA=mean(X(tempA,:));
    normal_temp=[];
    Xt=X(tempA,:);
    clear dis
    for j=1:length(tempA)
        dis(j)=pdist2(mean_tempA,Xt(j,:),'Euclidean');
    end
    [sx_dis,sy_dis]=sort(dis);
        temp=find(sx_dis<biaozhun2);
        choose=sy_dis(temp);
        normal=[normal;tempA(choose)];
end
normal=unique(normal);

%%
[m,n,b]=size(hsi);
X = reshape(hsi,m*n,b);
y=reshape(hsi_gt,m*n,1);
y=reshape(hsi_gt,m*n,1);
mask=zeros(m*n,1);
mask(anomaly)=1;
mask(normal)=2;
mask=reshape(mask,m,n);

figure;
imshow(mask,[])
mask=reshape(mask,m*n,1);
%save  HYDICE_urban_resize X y mask


