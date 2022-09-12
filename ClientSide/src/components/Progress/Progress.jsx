import React, { Component } from 'react'
import './Progress.css'

export default function Progress(props) {
  
  return (
    <div className="ProgressBar">
      <div
        className="Progress"
        style={{ width: props.progress + '%' }}
      />
    </div>
  )

}

