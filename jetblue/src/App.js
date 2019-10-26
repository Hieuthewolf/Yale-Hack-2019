import React from 'react';
import './App.css';
import Review from "./class/Review"
import {
  Route,
  NavLink,
  HashRouter
} from "react-router-dom";

import Home from "./class/Home"
import Contact from "./class/Contact"

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
      <HashRouter>
        <div className = "container">
          {reviewItems}
        </div>
        <div>
          <ul className="navigation-header">
            <li><NavLink to="/">Home</NavLink></li>
            <li><NavLink to="/contact">Contact</NavLink></li>
          </ul>
          <div className="content">
            <Route exact path="/" component={Home}/>
            <Route path="/contact" component={Contact}/>
          </div>
        </div>
      </HashRouter>
    );
  }
}

export default App;
