import React from 'react';
import './App.css';
import Review from "./class/Review";
import Navigation from './class/Navigation';

class App extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      reviews: [
        {title: "mewqmeqwmwqmewqmewqdsksnksdjnfafjlflfaafll", text: '23213'},
      ],
      display: 'home',
      filters: [],
    };
  }

  render() {
    const reviewItems = this.state.reviews.map((d) => <Review title={d.title} text={d.text} />)
    console.log(reviewItems);
    return (
      <div>
        <div>{Navigation()}</div>
        <div>
          {reviewItems}
        </div>
      </div>
    );
  }
}

export default App;
