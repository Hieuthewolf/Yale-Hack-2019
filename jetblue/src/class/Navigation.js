import React from 'react';
import Tabs from 'react-bootstrap/Tabs';
import Tab from 'react-bootstrap/Tab';

function Navigation(props) {
    return (
        <Tabs bg='dark' variant='dark'>
            <Tab eventKey='home' title='home'>
                Home
            </Tab>
            <Tab eventkey='bookmarks' title='bookmarks'>
                Bookmarks
            </Tab>
        </Tabs>
    )
}

export default Navigation;