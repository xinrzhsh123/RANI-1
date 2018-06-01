function Image_S=sparate_tou_sig_nuclei(path,name)
    name=char(name);
    I_CCNN=double(imread([path name]));
    if max(I_CCNN(:))>1
        I_CCNN=I_CCNN./max(I_CCNN(:));
    end
      
    I_siez=size(I_CCNN);
    I=double(I_CCNN);
    g=(I>=0.5);

    
    L = bwlabel(g, 8);
    stats=regionprops(L,'All');
    idx_nonsmallsize = find([stats.Area] >50);  
    stats=stats(idx_nonsmallsize);

    
    idx_size = find([stats.Area] > 450);  %set06: 450
    
    A=[stats.Area];
    P=[stats.Perimeter];

    Major_A=[stats.MajorAxisLength];
    Minor_A=[stats.MinorAxisLength];
    idx_long=find(Major_A./Minor_A>1.8);% long  set6:1.8
    idx_size_2=find([stats.Area] > 70);        %set6:70
    idx_long=intersect(idx_size_2,idx_long);
    
    
    idx_concavity=find([stats.ConvexArea]-[stats.Area]>=15);% solidity     set6:0.96
    idx_size_3=find([stats.Area] > 150);               

    idx_touching=union(idx_long,idx_concavity);
    idx_touching=union(idx_touching,idx_size);
    
    
    T_touching=zeros(I_siez);
    
    for i=1:length(idx_touching)
        T_touching(stats(idx_touching(i)).PixelIdxList)=1;
    end
    
    T_touching(T_touching>0)=1;  
    T_single=g-T_touching;  %TT: mask of isolated


    Image_touching=T_touching.*double(I_CCNN);
    Image_single=T_single.*double(I_CCNN);

    Image_S(1,:,:)=Image_single;
    Image_S(2,:,:)=Image_touching;

 
    




