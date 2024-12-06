big=uint16(rand([512,512,3])*((2^16)-1));
writetif_color(big,'tester.tif');