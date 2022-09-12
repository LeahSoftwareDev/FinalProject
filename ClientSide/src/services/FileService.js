import axios from "axios"; 
class FileService {

    get_wav_file() {
     
        return axios({
            url: 'http://127.0.0.1:5000/download',
            method: 'GET',
            responseType: 'blob',
        }).then((response) => {
            const url = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', 'audioplayback.wav');
            document.body.appendChild(link);
            console.log(link)
            link.click();
        });
    }
    } export default new FileService