import React from 'react';
import './App.css';
import Review from "./class/Review"

class App extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      reviews: [<Review title = "mewqmeqwmwqmewqmewqdsksnksdjnfafjlflfaafll" text = "23213" />],
      display: 'home',
      filters: [],
    };
  }

  render() {
    const reviewItems = this.state.reviews.map((d) => d)
    console.log(reviewItems);
    return (
      <div>
        {reviewItems}
      </div>
    );
  }
}

export default App;
