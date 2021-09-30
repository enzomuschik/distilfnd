############ DISTILFND-WEB-APPLICATION ############

Autor: Enzo Muschik
Datum: 29.09.2021
Masterarbeit: Explainable detection of fake news with Deep Learning
Universität: FernUniversität in Hagen, Fakultät für Mathematik und Informatik

###################################################
Dieser Ordner beinhaltet den Programmcode für folgende Module:

1. DistilFND-Webapplikation: distilfnd_web_app.py
2. DistilFND Class/Module: distilfnd.py
3. DataLoader Class/module: dataloader.py
4. DistilFND-Webapplikation Launcher: launch_web_app.bat

Anleitung zur Anpassung der Systemvariablen um mitgegebenen
python-3.8.10 Interpreter inklusive vorinstallierten Modulen 
in Windows-Command-Prompt zu nutzen:

1. Navigieren Sie zum initialen Ordner (zwei Ordner zurück von hier)
   und dann in den Ordner "python-3.8.10"
2. Kopieren Sie den Pfad in diesen Ordner "python-3.10.1", bspw.
   "E:\python-3.8.10"
3. Unter Windows: Suchen Sie nach "Edit the system environment variables"
   bzw. "Verwalten von Umegbungsvariablen" und selektieren
4. Klicken Sie auf "Environment variables..." bzw. "Umgebungsvariablen..."
   in Pop-Up Dialog
5. In erneutem Pop-Up-Dialog: Unter Systemvariablen die Variable "Path" suchen >
   selektieren und "New/Neu" anklicken
6. In finalem Pop-Up Dialog: Die Pfade "E:\python-3.8.10" und "E:\python-3.8.10\Scripts"
   als Umgebungsvariablen hinzufügen und alle Pop-Up-Dialog mit OK bestätigen

## Nun wird die Windows Command Prompt diese mit allen Modulen vorinstallierte
   Python Interpreter Instanz zum Start der DistilFND-Webapplikation nutzen!##

Anleitung zum Starten der DistilFND-Webapplikation:

1. Doppelklick auf "launch_web_app.bat" Datei.
2. Google Chrome Browser mit geladener DistilFND-Webapplikation wird sich
   automatisch laden.
3. Sobald fertig geladen, können mit der Sidebar (Pfeil oben links) und
   dem Sample-Drop-Down Menü insgesamt 12 Beispiel Reddit-Posts von insgesamt
   6 Klassen ausgesucht werden.
4. Mit dem "PREDICT POST" Button wird das trainierte DistilFND-Modell angesteuert
   und das Prognoseergebnis für einen gegebenes Sample erscheint in der rechten Spalte.




