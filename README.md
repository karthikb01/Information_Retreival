Information retreival System 
-------------------------------------------------------------------------------------------------------------------------------

Information retreival System is a python based web application to retrieve releveant information from the internet for a query.

-------------------------------------------------------------------------------------------------------------------------------

## Prerequisites

Before you continue, ensure you have met the following requirements:
* Make sure you have an active internet connection.
* You have installed the latest version of python.
* You have a web browser installed and is up to date.
* Required python modules: sqlite3, nltk, flask, easygui, sklearn. 
* To install any of the modules use: pip --install 'module-name'

--------------------------------------------------------------------------------------------------------------------------------

## Operating instructions

1. Executing the main_last.py script:
	-> Double click on the "main_last.py".
	-> This will open the command prompt which will contain a url something like: "http://127.0.0.1:5000".
	-> Copy this url and paste it in the browser of your choice.

2. Executing using the command prompt:
	-> Open command prompt in the "Project" folder.
	-> Execute the main_last.py script using the command python -u "main_last.py".
	-> Copy the url that appears into your web browser.

--------------------------------------------------------------------------------------------------------------------------------

## This distribution package contains the following files:

  main_last.py  - Main python script.
  templates     - HTML templates for different pages.
  text_files    - .txt files created by the application that temporarily stores the ouput.
  key.txt       - stores the search query temporarily.
  multisearch1.db        - sql lite file for database.
  stopwords.txt        - contains the list of stop wordsused in query pre-processing.

  License.txt   - License information
  readme.txt    - This file
  
--------------------------------------------------------------------------------------------------------------------------------

## Features

The main features of 7-Zip: 
 	-> Provides relevant information along with latest news.
	-> User can also choose to get the latest news related to different topics like business, technology, sports etc.
	-> Works best for single word queries of latest topics and trends.

--------------------------------------------------------------------------------------------------------------------------------

## BUGS

These are the possible bugs:
	-> Unable to connect:
		Make sure that your internet connection is stable.
	-> Key error:
		This is an api error. This can happen when the key has run out of requests for a day. Try to change the api key 
		in the code by commenting on of the keys and un-commenting the other.

--------------------------------------------------------------------------------------------------------------------------------

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


---------------------------------------------------------------------------------------------------------------------------------

## Contributors

Karthik B : https://github.com/karthikb01 
Aniketh B R: https://github.com/aniketh_br
Ajay Krishna M: https://github.com/ajay_m
Amogh Banare: https://github.com/amogh_banare

---------------------------------------------------------------------------------------------------------------------------------

## Acknowledgements

It is my great pleasure to acknowledge the assistance and contributions of all the people who helped me to make this application
successful. The projecct would not have been so successful without the dedicated assistance given by these individuals.

I wish to express my deepest gratitude to project guide Prof. Pavithra D S, Department of Computer Science and Engineering, 
CEC, Mangalore, for the guidance and suggestions.

---------------------------------------------------------------------------------------------------------------------------------

## Contact information

If you have any problems, questions, ideas or suggestions, please contact us using one of these ways:
karthikbannur@gmail.com
anikethbr73@gmail.com

---------------------------------------------------------------------------------------------------------------------------------

## License

Copyright (c) [2023] [Information_Retrieval_System]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files 
(the "Software"), to deal in the Software without restriction, including without limitation the rightsto use, copy, modify, 
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


---------------------------------------------------------------------------------------------------------------------------------
End of document
