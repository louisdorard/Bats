% http://www.zaphu.com/2007/09/13/how-to-customize-and-improve-matlab-figures-for-publication/

fh = figure(1); % returns the handle to the figure object
set(fh, 'color', 'white'); % sets the color to white
axes('FontSize', 12)

ColorSet = varycolor(3);
set(gca, 'ColorOrder', ColorSet);

hold all;
nit = 10;
range = 1:(nit/10):nit;

plot(range, sqrt(range), 'LineWidth', 2);
plot(range, range, 'LineWidth', 2);
plot(range, range.^2, 'LineWidth', 2);

%legend('Square root','Linear','Quadratic');

xlabel( 'Number of iterations', 'FontSize', 14, 'FontWeight', 'bold');
ylabel( 'Regret', 'FontSize', 14, 'FontWeight', 'bold', 'Rotation', 90 );

% print(fh, '-dpng', 'figure.png');
% saveas(fh, 'FruitflyPopulation', 'epsc');
