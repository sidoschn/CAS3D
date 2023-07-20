getDimensions(imgWidth, imgHeight, channels, imgSlices, frames);

//imgWidth = getWidth();
//imgHeight = getHeight();

frequency = 0.5;
angleDeg = 40;
angle = Math.toRadians(angleDeg);
ox = 256;
oy = 256;
oz = 256;

a = 1;
b = 0;
c = 128;

for (z = 0; z<imgSlices;z++){
	setSlice(z+1);
	for (x = 0; x < imgWidth; x++) {
		for (y = 0; y <imgHeight; y++) {
	
		//setPixel(x, y, (((sin(frequency*(x-(y*tan(angle))))+1)/2)  ));
		//setPixel(x, y, ((sin(frequency*(x*cos(angle))+frequency*(y*sin(angle))))+1)/2);
		//setPixel(x, y, sin(sqrt(((x-ox)*(x-ox))+((y-oy)*(y-oy))+((z-oz)*(z-oz)))));
		setPixel(x, y, a*exp( -(((x-ox)*(x-ox))+((y-oy)*(y-oy))+((z-oz)*(z-oz)))/(2*c*c) ));
		}
	}
}
updateDisplay();