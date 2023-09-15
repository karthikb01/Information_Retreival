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

    fetch('../text_files/newsApi.txt')
        .then(response => response.text())
        .then(data => {
            fileContents = data;
            document.getElementById('news').innerText = fileContents;
        })
        .catch(error => console.error(error));

}