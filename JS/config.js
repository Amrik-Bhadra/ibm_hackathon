const mysql = require('mysql');
const express = require('express');
const bodyparser = require('body-parser');
const path = require('path');
const nodemailer = require('nodemailer');
const randomstring = require('randomstring');
const session = require('express-session');
const ejs = require('ejs');
const fs = require('fs');

module.exports = {
    mysql,
    express,
    bodyparser,
    path,
    nodemailer,
    randomstring,
    session,
    ejs,
    fs
};
