import './App.css';
import React from "react";
import ReactDOM from "react-dom";
import UploadComponent from "./components/Upload/Upload";
import 'bootstrap/dist/css/bootstrap.css'



class App extends React.Component {
  state = {
    upload: {
      pictures: [],
      maxFileSize: 5242880,
      imgExtension: [".jpg", ".png"],
      defaultImages: []
    }
  };

  handleChange = files => {
    const { pictures } = this.state.upload;
    console.warn({ pictures, files });

    this.setState(
      {
        ...this.state,
        upload: {
          ...this.state.upload,
          pictures: [...pictures, ...files]
        }
      },
      () => {
        console.warn("It was added!");
      }
    );
  };


  render() {
    return (
      <div className="App">
        <div className="gap"></div>
        <br />
        <div className=' App-header'>
          <UploadComponent className='upload'
            {...this.state.upload}
            handleChange={this.handleChange}
          />
        </div>
      </div>
    );
  }
}

const rootElement = document.getElementById("root");
ReactDOM.render(<App />, rootElement);