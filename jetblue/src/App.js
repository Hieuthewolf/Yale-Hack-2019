import React from 'react';
import logo from './logo.svg';
import './App.css';

class Review extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      title: 'filler title',
      text: 'filler text',
      rating: 'filler rating',
      source: 'filler source',
    };
  }

  render() {
    return (
      <div className='review'>
        <h3>{this.state.title}</h3>
        <p>{this.state.text}</p>
        <p>{this.state.source}</p>
      </div>
    );
  }
}

class App extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      reviews: [],
      display: 'home',
      filters: [],
    };
  }

  render() {
    return (
      <div className='header'>JetBlue Reviews</div>
    );
  }
}

export default App;
