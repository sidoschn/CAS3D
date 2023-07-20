imgWidth = getWidth();
imgHeight = getHeight();

frequency = 5;
variation = 0.2;
angleDeg = 0;
angle = Math.toRadians(angleDeg);
randArray = newArray(imgWidth);

//currentWavelength = frequency-(random*variation);
currentWavelength =  32;
wavelengthSum = currentWavelength;

for (x = 0; x < imgWidth; x++) {
	
	for (y = 0; y <imgHeight; y++) {

	//setPixel(x, y, (((sin(frequency*(x-(y*tan(angle))))+1)/2)  ));
	//setPixel(x, y, ((sin((1/currentWavelength)*(x*cos(angle))+(1/currentWavelength)*(y*sin(angle))))+1)/2);
	setPixel(x, y, (sin(x/(currentWavelength*(currentWavelength/x)))));
	}
	
}

updateDisplay();

//
//if (x>wavelengthSum){
//		currentWavelength = ((frequency+random));
//		wavelengthSum = wavelengthSum+currentWavelength;
//		print(currentWavelength);
//	}