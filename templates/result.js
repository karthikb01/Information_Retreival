function display() {
    let query;

    fetch('../key.txt')
        .then(response => response.text())
        .then(data => {
            query = data;
            document.getElementById('key').innerText = query;
        })
        .catch(error => console.error(error));

    let fileContents;

    fetch('../text_files/rankedNews.txt')
        .then(response => response.text())
        .then(data => {
            fileContents = data;
            document.getElementById('news').innerText = fileContents;
            if(fileContents != undefined){
                document.getElementById('news').style.display = "block"
            }
        })
        .catch(error => console.error(error));

  

    let wikiContent;

    fetch('../text_files/aSummary.txt')
        .then(response => response.text())
        .then(data => {
            wikiContent = data;
            document.getElementById('wiki').innerText = wikiContent;
            if(wikiContent != undefined){
                document.getElementById('wiki').style.display = "block"
            }
        })
        .catch(error => console.error(error));
  

}