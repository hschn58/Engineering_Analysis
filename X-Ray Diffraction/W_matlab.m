

%this program plots the x-ray diffraction (XRD) pattern for tungsten

%it also analyzes error with this method, such as the sample surface
%being a bit too high or low in the XRD machine

%it also analyzes a source of error associated with the absorption of
%x-rays. In calculation, it is assumed that x-rays do not penetrate the
%surface, but in reality they do slightly. 

%These two issues can lead to peaks occurring at a slightly different
%theta values than theory would expect


%get the file contents
file='out_path/W.txt';
Table=readtable(file);
Table=table2array(Table);



xdat=Table(:,2);
ydat=Table(:,3);

%plot the initial data 
[pks, locs] = findpeaks(ydat, 'MinPeakProminence', 700);

locs=locs(1:end-1);

xdat=xdat(1:locs(end)+100);
ydat=ydat(1:locs(end)+100);

xpeaks=xdat(locs);
pks=pks(1:end-1);

plot(xdat,ydat, color='blue')
hold on;

%plot the peaks
scatter(xpeaks,pks, 'r')

strhkl=["(110)", "(200)", "(211)", "(220)", "(310)", "(222)", "(123)"];

for i=1:1:length(xpeaks)
    text(xpeaks(i), pks(i)+1100, strhkl(i), 'Rotation', 270);
end

xlabel('2$\theta$ ($^\circ$)', 'Interpreter', 'latex', 'FontSize',16);
ylabel('Intensity (counts)', 'FontSize',16);
grid on;


%save the diffraction pattern figure
print(gcf, 'out_path/W_pattern.png', '-dpng', '-r900');









%find the d values

%x-ray wavelength is 1.5406 Å
lambda=1.5406;

d=(lambda./(2.*sind(xdat(locs)*0.5)));


hkl=[1 1 0; 2 0 0; 2 1 1; 2 2 0; 3 1 0; 2 2 2; 1 2 3];

hkl2=[];
for i=1:1:length(pks)
    hkl2=[hkl2,(sum(hkl(i,:).^2).^(0.5))];
end

%find the a_0 values
a_0=[];
for i=1:1:length(pks)
    a_0=[a_0,d(i).*hkl2(i)];
end

%find the cos(theta)^2 values
ctheta_2=[];
theta_data=xdat(locs)*0.5;

for i=1:1:length(pks)
    ctheta_2=[ctheta_2,cosd(theta_data(i))^2];
end


%plot theta_^2 vs a_0

figure;
scatter(ctheta_2,a_0,'Marker', 'o', 'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'r', 'SizeData', 50)
hold on;

xlabel('$(cos(2\theta))^2$','Interpreter', 'latex','FontSize',16);
ylabel('Lattice Parameter (Å)', 'FontSize', 16);


% Use polyfit to fit a line of best fit (linear regression)
degree = 1; % Linear regression (degree 1)
coefficients = polyfit(ctheta_2, a_0, degree);

% Generate y-values for the line of best fit
y_fit = polyval(coefficients, ctheta_2);

% Plot the line of best fit
plot(ctheta_2, y_fit, 'r', 'LineWidth', 2);

grid on;


print(gcf, 'out_path/W_pattern_a0s.png', '-dpng', '-r900');


%tungsten lattice parameter is 3.165A


a_0_1=coefficients(2);

%make the new thetas
t_new=[];
for i=1:1:length(pks)
    
    t_new=[t_new, asind((hkl2(i)*lambda)/(2*a_0_1)).*2];
end

clc

% disp(hkl2)
% disp(t_new)
%negative half a millimeter
t_new=transpose(t_new);

delta_t = xpeaks - t_new;

% disp(xpeaks)
% disp(transpose(t_new))

%plot of delta vs theta obtained from the initial dataset
%disp(sin(xpeaks))
%disp(delta_t)

figure;
scatter(sind(xpeaks), delta_t,'MarkerFaceColor', 'blue' )
xlabel('$\sin(2 \theta_{b})$','Interpreter', 'latex', 'FontSize', 16);
ylabel('$\triangle 2 \theta_{\alpha}$','Interpreter', 'latex', 'FontSize', 16);
hold on;

% Use polyfit to fit a line of best fit (linear regression)
degree = 1; % Linear regression (degree 1)
coefficients = polyfit(sind(xpeaks), delta_t, degree);


% Generate y-values for the line of best fit
y_fit2 = polyval(coefficients, sind(xpeaks));

% Plot the line of best fit
plot(sind(xpeaks), y_fit2, 'blue', 'LineWidth', 2);


%s determination

%s= ((delta_avg)*pi*R*sin(a)/(180*sin(2theta))

%delta_t: degrees
%2theta: degrees
%a: degrees


a=12.7;
R=180; %mm

s= (-1*(delta_t)*pi*R*sind(a))./(180*sind(t_new));

%s = mean(s)= -0.293787 millimeters

myFloat = mean(s);
myString = sprintf('The value of s is approximately: %f', myFloat);
disp(myString);


%angular shift:
%at 8000.37eV, the attenuation length for W is 3.10893 microns
%thus, mu is 1/(3.10893 (in cm)) = 3216.54

mu=3216.54;

delta= 180 ./(pi*mu*180.*(tan(t_new./2)+cot(t_new-a)));

%do not expect the error associated with x-ray absorption
%to be significant since mu is greater than 100. 


T=[xpeaks, pks, d, hkl2.^(2)', a_0', delta_t, s, delta];

Table = array2table(T, 'VariableNames', {'2Theta (degrees)','Intensity (counts)','d_hkl','hkl^2','a_0', 'Height imacted shift (deg)', 'Height Error', 'Transparency-Induced shift (deg)'});

writetable(Table, 'out_path/peaks.csv');



