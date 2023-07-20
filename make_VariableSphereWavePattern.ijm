getDimensions(imgWidth, imgHeight, channels, imgSlices, frames);

//imgWidth = getWidth();
//imgHeight = getHeight();

frequency = 0.5;
angleDeg = 40;
angle = Math.toRadians(angleDeg);
ox = 256;
oy = 256;
oz = 0;

for (z = 0; z<imgSlices;z++){
	setSlice(z+1);
	for (x = 0; x < imgWidth; x++) {
		for (y = 0; y <imgHeight; y++) {
	
		//setPixel(x, y, (((sin(frequency*(x-(y*tan(angle))))+1)/2)  ));
		//setPixel(x, y, ((sin(frequency*(x*cos(angle))+frequency*(y*sin(angle))))+1)/2);
		r = sqrt(((x-ox)*(x-ox))+((y-oy)*(y-oy))+((z-oz)*(z-oz)));
		setPixel(x, y, cos(r*0.1*r));
		}
	}
}
updateDisplay();