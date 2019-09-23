const fs = require('fs');
const http = require('http');
const express = require('express');
var path = require('path');

const app = express();
const port = 3000;

app.use(express.static(path.join(__dirname)));

app.get('/', (req, res) => {
    res.writeHead(200);

    if (req.method == 'POST') {
        console.log('POST request');
        req.on('data', function (chunk) {
            var formdata = chunk.toString();
            print('formdata: ' + formdata);
        })
    }

    if (req.url == '/favicon.ico') {
        console.log('FAVICON REQUESTED');
        res.writeHead(200, { 'Content-Type': 'image/x-icon' });
        res.end();
        return;
    }
    if (req.url == '/') {
        req.url = '/index.html';
        console.log(__dirname + req.url);
    }
    res.end(fs.readFileSync(__dirname + req.url));

});

app.post('/svm', function (req, res) {
    req.on('data', function (chunk) {
        let runPy = new Promise(function (success, nosuccess) {
            const { spawn } = require('child_process');
            const pyProgram = spawn('python3', ['./../predict.py', chunk]);

            pyProgram.stdout.on('data', function (data) {
                console.log('success!')
                success(data);
            });
            pyProgram.stderr.on('data', (data) => {
                console.log('no success')
                nosuccess(data);
            });
        }).then(function (fromRunPy) {
            console.log('node js result: ' + fromRunPy.toString());
            res.end(fromRunPy);
        }).catch(function (error) {
            console.log('error!')
            console.log(error);
        });
    });
})


app.listen(port, function () {
    console.log("Site running on localhost:3000")
});
