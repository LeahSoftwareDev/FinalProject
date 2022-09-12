import React, { Component } from 'react'
import { useRef } from 'react';
import { useState } from 'react'
import './Dropzone.css'

export default function Dropzone(props){
 const [highLight,setHighLight]=useState(false);
 const fileInputRef=useRef();

  const openFileDialog=()=> {
    if (props.disabled) return
    fileInputRef.current.click()
  }

  // event on the input change and get file
  const onFilesAdded=(evt)=> {
    if (props.disabled) return
  
    const files = evt.target.files
    if (props.onFilesAdded) {
        
      const array = fileListToArray(files)
      props.onFilesAdded(array)
    }
  }

  const onDragOver=(evt) =>{
    evt.preventDefault()
    if (props.disabled) return
    setHighLight(true);
  }

  const onDragLeave=()=> {
    setHighLight(false);
  }
  
  const onDrop=(event) =>{
    event.preventDefault()

    if (props.disabled) return

    const files = event.dataTransfer.files
    if (props.onFilesAdded) {
      const array = fileListToArray(files)
      props.onFilesAdded(array)
      
    }
    setHighLight(false);
  }

  const fileListToArray=(list) =>{
    const array = []
    for (var i = 0; i < list.length; i++) {
      array.push(list.item(i))
    }
    return array
  }

    return (
      <div
        className={`Dropzone ${highLight ? 'Highlight' : ''}`}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        onClick={openFileDialog}
        style={{ cursor: props.disabled ? 'default' : 'pointer' }}
      >
        <input
          ref={fileInputRef}
         
          className="FileInput"
          type="file"
          onChange={onFilesAdded}
        />
        <img
          alt="upload"
          className="Icon"
          src="baseline-cloud_upload-24px.svg"
        />
        <span>Upload Files</span>
      </div>
    )
  
}

