function [level]=triangle_th2(lehisto,num_bins)
%     Triangle algorithm
%     This technique is due to Zack (Zack GW, Rogers WE, Latt SA (1977), 
%     "Automatic measurement of sister chromatid exchange frequency", 
%     J. Histochem. Cytochem. 25 (7): 741�53, )
%     A line is constructed between the maximum of the histogram at 
%     (b) and the lowest (or highest depending on context) value (a) in the 
%     histogram. The distance L normal to the line and between the line and 
%     the histogram h[b] is computed for all values from a to b. The level
%     where the distance between the histogram and the line is maximal is the 
%     threshold value (level). This technique is particularly effective 
%     when the object pixels produce a weak peak in the histogram.

%     Use Triangle approach to compute threshold (level) based on a
%     1D histogram (lehisto). num_bins levels gray image. 

%     INPUTS
%         lehisto :   histogram of the gray level image
%         num_bins:   number of bins (e.g. gray levels)
%     OUTPUT
%         level   :   threshold value in the range [0 1];
% 
%     Dr B. Panneton, June, 2010
%     Agriculture and Agri-Food Canada
%     St-Jean-sur-Richelieu, Qc, Canad
%     bernard.panneton@agr.gc.ca


%   Find maximum of histogram and its location along the x axis
    lehisto=double(lehisto);
    num_bins=double(num_bins);
    [h,xmax]=max(lehisto);
   
    
%   Find location of first and last non-zero values.
    indi=find(lehisto>0);
    fnz=max(indi(1)-1,1);
    lnz=min(indi(end)+1,num_bins);

%   Pick side as side with longer tail. Assume one tail is longer.
    lspan=xmax-fnz;
    rspan=lnz-xmax;
    if rspan>lspan  % then flip lehisto
        lehisto=fliplr(lehisto);
        a=num_bins-lnz+1;
        b=num_bins-xmax+1;
        isflip=1;
    else
        lehisto=lehisto;
        isflip=0;
        a=fnz;
        b=xmax;
    end
    
%   Compute parameters of the straight line from first non-zero to peak
%   To simplify, shift x axis by a (bin number axis)
    nx=h;
    ny=a-b;
    d=sqrt(nx^2+ny^2);
    nx=nx/d;
    ny=ny/d;
    d=nx*a+ny*lehisto(a);
    
    
%   Compute distances
    x1=a+1:b;
    y1=lehisto(x1);
    L=x1*nx+y1*ny-d;
    
%   Obtain threshold as the location of maximum L.    
    level=find(max(L)==L)+a;
    
    
%   Flip back if necessary
    if isflip
        level=num_bins-level+1;
    end
    level=level-1;%index to graylevel
    
    
% figure
% bar(lehisto); xlim([0 num_bins]);
% line([xmax h],[lnz 0],'Color','r','LineWidth',2);
% line([le le],[0 max(ylim)],'Color','b','LineWidth',2);
% hold on;
% plot(10,50,'y');  %Dummy point to make a 2 line entry in legend
% line([126 126],[0 max(ylim)],'Color','g','LineWidth',2);
% hl=legend('Data','Base line used in Triangle Method','Threshold by Triangle Method','(see triangle_th.m for details)','Threshnold by graythresh.m');
% set(hl,'Interpreter','none');
    
   