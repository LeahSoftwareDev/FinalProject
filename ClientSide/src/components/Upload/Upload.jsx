import React, { useState } from "react";
import ImageUploader from "react-images-upload";
import axios from "axios"
import "./Upload.css"
import Download from '../Download/Download';
import Audio from '../Audio/Audio';

export default function UploadComponent(props) {

  const [file, setFile] = useState([]);
  const [uploadProgress, setUploadProgress] = useState({});
  const [SuccessfullUploaded, setSuccessfullUploaded] = useState({});
  const [Result, setResult] = useState({});
  const [Progress, setProgress] = useState({});
  const [Upload, setUpload] = useState({});
  const [showUpload, setShowUpload] = useState(true)
  const [image, setImage] = useState({});


  const onDrop = (pictureFiles, pictureDataURLs) => {
    setFile(file.concat(pictureFiles))
    const newImagesUploaded = pictureDataURLs.slice(
      props.defaultImages.length
    );
    console.warn("pictureDataURLs =>", newImagesUploaded);
    props.handleChange(newImagesUploaded);
  };

  const uploade_file = () => {

    const formData = new FormData();

    formData.append('image', file[0]);
    setImage(formData)
    console.log(formData)
    axios.post('http://127.0.0.1:5000/upload', formData, {

      onUploadProgress: (progressEvent) => {
        //how much time need finish
        if (progressEvent.lengthComputable) {
          const copy = { ...uploadProgress };
          copy[file[0].name] = {
            state: "pending",
            percentage: (progressEvent.loaded / progressEvent.total) * 100
          };
          setUploadProgress(copy);
        }
      },
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
      .then(response => {
        setShowUpload(false)
        const copy = { ...uploadProgress };
        copy[file[0].name] = { state: "done", percentage: 100 };
        setUploadProgress(copy);
        setSuccessfullUploaded(true)
        setResult(response.data)
        console.log(response.data)
        return true;
      })
      .catch((error) => {
        if (error.response) {
          console.log(error.response)
          console.log(error.response.status)
          console.log(error.response.headers)
          const copy = { ...uploadProgress };
          copy[file[0].name] = { state: "error", percentage: 0 };
          setUpload(true)
          Progress(copy)
          return false;
        }
      })
  }

  return (
    <div>
        <div className="upload">
          <ImageUploader className="fileUploader c"
            withIcon={false}
            withLabel={false}
            withPreview={true}
            buttonText={"Upload sheet"}
            fileSizeError={"File size is too big!"}
            fileTypeError={"This extension is not supported!"}
            onChange={onDrop}
            imgExtension={props.imgExtension}
            maxFileSize={props.maxFileSize}
            buttonClassName='fileContainer chooseFileButton'
            style={{ fill: "#FFFFFF" }}
          />
        </div>
          {showUpload ?
      <div className="confirm">
        <button type="button" className="btn btn-outline-info" onClick={uploade_file}>Confirm upload</button>
      </div>:
      <div>
        <Audio className='audio' />
        <Download className='download' />
      </div>
      }
    </div>
  )
}
