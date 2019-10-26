import React from 'react';

class Review extends React.Component {
    constructor(props) {
      super(props);
  
      this.state = {
        title: props.title,
        text: props.text,
        rating: props.rating,
        source: props.source
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

export default Review;