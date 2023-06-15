# How to use bioimageio model in PartSeg


`This tutorial requires full implement of interface for screenshots`

1. Ładowanie danych do PartSega przez przeciągniecie
2. wybranie z listy dostępnych metod segmentacji metodę z tej paczki
3. wybranie modelu z dysku
4. ustawienie parametrów rekonstrukcji
5. uruchomienie segmentacji
6. analiza wyników

Powiedzenie, że można uzyć w batchprocessingu i ostrzerzenie o tym, że jeżeli instalacja pytorcha używa cudy,
to trzeba uważać na pamięć GPU.

## Prepare data

To load the data user could drag and drop the file to PartSeg window or use `File->Open` menu.
In this tutorial we will use the image from [Zenodo](link_here).

![Main window view](images/PartSeg_main_window.png)
