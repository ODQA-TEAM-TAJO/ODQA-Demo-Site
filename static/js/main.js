/* get response script */
function getBotResponse() {
    var rawText = $("#textInput").val();
    var userHtml = '<p class="userText"><span>' + rawText + '</span></p>';
    $("#textInput").val("");
    $("#chatbox").append(userHtml);
    document.getElementById('userInput').scrollIntoView({block: 'start', behavior: 'smooth'});
    var objDiv = document.getElementById("chatbox");
    objDiv.scrollTop = objDiv.scrollHeight;
    $.get("/get", { msg: rawText }).done(function(data) {
      var botHtml = '<img id="botImg" src="static/image/group_tajo_1.png"><img/><p class="botText"><span>' + data + '</span></p>';
      $("#chatbox").append(botHtml);
      document.getElementById('userInput').scrollIntoView({block: 'start', behavior: 'smooth'});
      var objDiv = document.getElementById("chatbox");
      objDiv.scrollTop = objDiv.scrollHeight;
    });
  }
  $("#textInput").keypress(function(e) {
      if ((e.which == 13) && document.getElementById("textInput").value != "" ){
          getBotResponse();
          var objDiv = document.getElementById("chatbox");
          objDiv.scrollTop = objDiv.scrollHeight;
      }
  });
  $("#buttonInput").click(function() {
      if (document.getElementById("textInput").value != "") {
          getBotResponse();
          var objDiv = document.getElementById("chatbox");
          objDiv.scrollTop = objDiv.scrollHeight;
      }
  })