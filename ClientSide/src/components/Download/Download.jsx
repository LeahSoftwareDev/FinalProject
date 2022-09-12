import 'bootstrap/dist/css/bootstrap.min.css';
import FileService from "../../services/FileService";
import './Download.css'
import React from "react"; 


export default function Download() {
  const downloadFile = () => {
    FileService.get_wav_file()
  }

  return (
    <div>
      <button className='btn btn-outline-info' onClick={downloadFile}> Download </button>
    </div>
  );
}